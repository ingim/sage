pub mod conv;
pub mod core;
pub mod gemm;
pub mod map;
pub mod metric;
pub mod nn;
pub mod rand;
pub mod reduce;
use crate::error::Error;
use crate::ops::core::LayoutOperator;
use crate::ops::map::{MapOperator, NullaryMapOperator, StackElement, VariadicMapOperator};
use crate::ops::reduce::ReduceOperator;
use crate::session::context::{CachedAccess, Context};
use crate::session::memory::MemoryError;
use crate::var::Var;
use itertools::Itertools;
use smallvec::{smallvec, SmallVec};
use std::cmp;
use std::fmt::{Debug, Formatter};

use crate::tensor::{Tensor, TensorDesc};

// Used as an optimization hint for the runtime compiler
#[derive(Clone, Debug)]
pub enum Category {
    Map(MapOperator),
    Reduce(ReduceOperator),
    Contract,
    Layout(LayoutOperator),
    Other,
}

pub enum Operator {
    Nullary(Box<dyn NaryOperator<0>>),
    Unary(Box<dyn NaryOperator<1>>),
    Binary(Box<dyn NaryOperator<2>>),
    Ternary(Box<dyn NaryOperator<3>>),
    Variadic(Box<dyn VariadicOperator>),
}
//
// pub struct Desc<const N: usize> {
//     pub input: [TensorDesc; N],
//     pub output: TensorDesc,
//     pub cache: CachedAccess,
// }
//
// impl<const N: usize> Desc<N> {
//     pub fn new(input: [TensorDesc; N], output: TensorDesc) -> Self {
//         Desc {
//             input,
//             output,
//             cache: CachedAccess::new(),
//         }
//     }
// }
//
// impl<const N: usize> Clone for Desc<N> {
//     fn clone(&self) -> Self {
//         Desc {
//             input: self.input.clone(),
//             output: self.output.clone(),
//             cache: CachedAccess::new(),
//         }
//     }
// }
//
// impl<const N: usize> Debug for Desc<N> {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         write!(
//             f,
//             "input: {:?}, output: {:?}, cached: {:?}",
//             &self.input,
//             &self.output,
//             !self.cache.is_empty()
//         )
//     }
// }

pub trait NaryOperator<const N: usize>: Debug {
    fn input(&self) -> &[TensorDesc; N];
    fn output(&self) -> &TensorDesc;

    fn grad(&self, x: [&Var; N], y: &Var, gy: &Var) -> [Option<Var>; N];
    fn compute(&self, x: [&Tensor; N], ctx: &mut Context) -> Result<Tensor, Error>;

    fn cat(&self) -> Category {
        Category::Other
    }
}

pub trait VariadicOperator: Debug {
    fn input(&self) -> &[TensorDesc];
    fn output(&self) -> &TensorDesc;

    fn grad(&self, x: &[Var], y: &Var, gy: &Var) -> Vec<Option<Var>>;
    fn compute(&self, x: &[Tensor], ctx: &mut Context) -> Result<Tensor, Error>;

    fn cat(&self) -> Category {
        Category::Other
    }
}

impl Operator {
    pub fn fuse(opr1: &Operator, opr2: &Operator, idx: usize) -> Option<Operator> {
        match (opr1.cat(), opr2.cat()) {
            // map + map -> map
            (Category::Map(op1), Category::Map(op2)) => {
                let mut input = Vec::with_capacity(op1.input().len() + op2.input().len() - 1);

                // if op1.input().len() != 1 && op1.input().len() != 2 {
                //     return None;
                // }

                let mut expr = vec![map::StackElement::Operator(op1.clone())];

                for i in 0..op1.input().len() {
                    if i == idx {
                        expr.push(StackElement::Operator(op2.clone()));
                        for j in 0..op2.input().len() {
                            input.push(op2.input()[j].clone());
                            expr.push(StackElement::Input(input.len() - 1));
                        }
                    } else {
                        input.push(op1.input()[i].clone());
                        expr.push(StackElement::Input(input.len() - 1));
                    }
                }

                // println!(
                //     "fused map! op{:?} + op{:?} (at: {:?}) -> op{:?}",
                //     op1.input().len(),
                //     op2.input().len(),
                //     idx,
                //     input.len()
                // );

                Some(Operator::Variadic(Box::new(VariadicMapOperator::new(
                    input,
                    op1.output().pristine(),
                    expr,
                ))))
            }

            // nullmap + expand -> nullmap
            (Category::Layout(op1), Category::Map(op2)) => {
                //return None;
                if let (LayoutOperator::Expand(op1), MapOperator::Nullary(op2)) = (op1, op2) {
                    Some(Operator::Nullary(Box::new(NullaryMapOperator::new(
                        TensorDesc::new(op1.output().extents(), op2.output.data_type()),
                        op2.map,
                    ))))
                } else {
                    None
                }
            }

            // Other fusion possibilities: Layout + Layout
            //
            (_, _) => None,
        }
    }

    pub fn compute(&self, x: &[Tensor], ctx: &mut Context) -> Result<Tensor, Error> {
        // check spec

        assert_eq!(x.len(), self.input().len());
        // println!("actual: {:?}", x.iter().map(|v| v.shape()).collect_vec());
        // println!(
        //     "expect: {:?}",
        //     self.input().iter().map(|v| v.shape()).collect_vec()
        // );
        // println!("{:?}", self.cat());

        assert!(x
            .iter()
            .zip(self.input().iter())
            .all(|(x, x_desc)| x.extents() == x_desc.extents()
                && x.data_type() == x.desc.data_type()));
        // TODO: check shape correspondance?
        match self {
            Operator::Nullary(opr) => opr.compute([], ctx),
            Operator::Unary(opr) => opr.compute([&x[0]], ctx),
            Operator::Binary(opr) => opr.compute([&x[0], &x[1]], ctx),
            Operator::Ternary(opr) => opr.compute([&x[0], &x[1], &x[2]], ctx),
            Operator::Variadic(opr) => opr.compute(x, ctx),
        }
    }

    pub fn arity(&self) -> usize {
        match self {
            Operator::Nullary(_) => 0,
            Operator::Unary(_) => 1,
            Operator::Binary(_) => 2,
            Operator::Ternary(_) => 3,
            Operator::Variadic(opr) => opr.input().len(),
        }
    }

    pub fn input(&self) -> &[TensorDesc] {
        match self {
            Operator::Nullary(_) => &[],
            Operator::Unary(opr) => opr.input(),
            Operator::Binary(opr) => opr.input(),
            Operator::Ternary(opr) => opr.input(),
            Operator::Variadic(opr) => opr.input(),
        }
    }

    pub fn output(&self) -> &TensorDesc {
        match self {
            Operator::Nullary(opr) => opr.output(),
            Operator::Unary(opr) => opr.output(),
            Operator::Binary(opr) => opr.output(),
            Operator::Ternary(opr) => opr.output(),
            Operator::Variadic(opr) => opr.output(),
        }
    }

    pub fn cat(&self) -> Category {
        match self {
            Operator::Nullary(opr) => opr.cat(),
            Operator::Unary(opr) => opr.cat(),
            Operator::Binary(opr) => opr.cat(),
            Operator::Ternary(opr) => opr.cat(),
            Operator::Variadic(opr) => opr.cat(),
        }
    }
}

impl Debug for Operator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Operator::Nullary(opr) => write!(f, "nullary.{:?}", opr),
            Operator::Unary(opr) => write!(f, "unary.{:?}", opr),
            Operator::Binary(opr) => write!(f, "binary.{:?}", opr),
            Operator::Ternary(opr) => write!(f, "ternary.{:?}", opr),
            Operator::Variadic(opr) => write!(f, "variadic.{:?}", opr),
        }
    }
}

pub struct Operation {
    opr: Operator,
    t_order: usize,
    x: SmallVec<[Var; 3]>,
}

impl Operation {
    pub fn nullary<O>(opr: O) -> Self
    where
        O: NaryOperator<0> + 'static,
    {
        Operation {
            opr: Operator::Nullary(Box::new(opr)),
            t_order: 0,
            x: smallvec![],
        }
    }

    pub fn unary<O>(opr: O, x: Var) -> Self
    where
        O: NaryOperator<1> + 'static,
    {
        Operation {
            opr: Operator::Unary(Box::new(opr)),
            t_order: x.t_order() + 1,
            x: smallvec![x],
        }
    }

    pub fn binary<O>(opr: O, x1: Var, x2: Var) -> Self
    where
        O: NaryOperator<2> + 'static,
    {
        Operation {
            opr: Operator::Binary(Box::new(opr)),
            t_order: cmp::max(x1.t_order(), x2.t_order()) + 1,
            x: smallvec![x1, x2],
        }
    }

    pub fn ternary<O>(opr: O, x1: Var, x2: Var, x3: Var) -> Self
    where
        O: NaryOperator<3> + 'static,
    {
        Operation {
            opr: Operator::Ternary(Box::new(opr)),
            t_order: cmp::max(cmp::max(x1.t_order(), x2.t_order()), x3.t_order()) + 1,
            x: smallvec![x1, x2, x3],
        }
    }

    pub fn variadic<O>(opr: O, x: Vec<Var>) -> Self
    where
        O: VariadicOperator + 'static,
    {
        let max_t_order = x
            .iter()
            .map(|a| a.t_order()) // each rank
            .max() // get max rank in parent gen
            .unwrap();

        let x = x.into_iter().collect();

        Operation {
            opr: Operator::Variadic(Box::new(opr)),
            t_order: max_t_order + 1,
            x,
        }
    }
}

impl Operation {
    pub fn t_order(&self) -> usize {
        self.t_order
    }

    pub fn input(&self) -> &[Var] {
        &self.x
    }

    pub fn opr(&self) -> &Operator {
        &self.opr
    }

    pub fn grad(&self, y: &Var, gy: &Var) -> Vec<Option<Var>> {
        match &self.opr {
            Operator::Nullary(_) => Vec::new(),
            Operator::Unary(opr) => opr.grad([&self.x[0]], y, gy).to_vec(),
            Operator::Binary(opr) => opr.grad([&self.x[0], &self.x[1]], y, gy).to_vec(),
            Operator::Ternary(opr) => opr
                .grad([&self.x[0], &self.x[1], &self.x[2]], y, gy)
                .to_vec(),
            Operator::Variadic(opr) => opr.grad(&self.x, y, gy),
        }
    }

    pub fn cat(&self) -> Category {
        self.opr.cat()
    }
}

impl Debug for Operation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.opr {
            Operator::Nullary(opr) => write!(f, "{:?}", self.opr),
            Operator::Unary(opr) => write!(f, "{:?}", self.opr),
            Operator::Binary(opr) => write!(f, "{:?}", self.opr),
            Operator::Ternary(opr) => write!(f, "{:?}", self.opr),
            Operator::Variadic(opr) => write!(f, "{:?}", self.opr),
        }

        // match &self.opr {
        //     Operator::Nullary(opr) => write!(f, "{:?}", self.opr),
        //     Operator::Unary(opr) => write!(f, "{:?}({:?})", self.opr, self.x[0]),
        //     Operator::Binary(opr) => write!(f, "{:?}({:?}, {:?})", self.opr, self.x[0], self.x[1]),
        //     Operator::Ternary(opr) => write!(
        //         f,
        //         "{:?}({:?}, {:?}, {:?})",
        //         self.opr, self.x[0], self.x[1], self.x[2]
        //     ),
        //     Operator::Variadic(opr) => write!(f, "{:?}({:?})", self.opr, self.x),
        // }
    }
}
