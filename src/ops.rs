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
use crate::var::Fun;
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

pub enum Composer {
    Nullary(Box<dyn Compose<0>>),
    Unary(Box<dyn Compose<1>>),
    Binary(Box<dyn Compose<2>>),
    Ternary(Box<dyn Compose<3>>),
    Variadic(Box<dyn VariadicCompose>),
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

pub trait Compose<const N: usize>: Debug {
    fn input(&self) -> &[TensorDesc; N];
    fn output(&self) -> &TensorDesc;

    fn grad(&self, x: [&Fun; N], y: &Fun, gy: &Fun) -> [Option<Fun>; N];
    fn compute(&self, x: [&Tensor; N], ctx: &mut Context) -> Result<Tensor, Error>;

    fn cat(&self) -> Category {
        Category::Other
    }
}

pub trait VariadicCompose: Debug {
    fn input(&self) -> &[TensorDesc];
    fn output(&self) -> &TensorDesc;

    fn grad(&self, x: &[Fun], y: &Fun, gy: &Fun) -> Vec<Option<Fun>>;
    fn compute(&self, x: &[Tensor], ctx: &mut Context) -> Result<Tensor, Error>;

    fn cat(&self) -> Category {
        Category::Other
    }
}

impl Composer {
    pub fn fuse(opr1: &Composer, opr2: &Composer, idx: usize) -> Option<Composer> {
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

                Some(Composer::Variadic(Box::new(VariadicMapOperator::new(
                    input,
                    op1.output().pristine(),
                    expr,
                ))))
            }

            // nullmap + expand -> nullmap
            (Category::Layout(op1), Category::Map(op2)) => {
                //return None;
                if let (LayoutOperator::Expand(op1), MapOperator::Nullary(op2)) = (op1, op2) {
                    Some(Composer::Nullary(Box::new(NullaryMapOperator::new(
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
            Composer::Nullary(opr) => opr.compute([], ctx),
            Composer::Unary(opr) => opr.compute([&x[0]], ctx),
            Composer::Binary(opr) => opr.compute([&x[0], &x[1]], ctx),
            Composer::Ternary(opr) => opr.compute([&x[0], &x[1], &x[2]], ctx),
            Composer::Variadic(opr) => opr.compute(x, ctx),
        }
    }

    pub fn arity(&self) -> usize {
        match self {
            Composer::Nullary(_) => 0,
            Composer::Unary(_) => 1,
            Composer::Binary(_) => 2,
            Composer::Ternary(_) => 3,
            Composer::Variadic(opr) => opr.input().len(),
        }
    }

    pub fn input(&self) -> &[TensorDesc] {
        match self {
            Composer::Nullary(_) => &[],
            Composer::Unary(opr) => opr.input(),
            Composer::Binary(opr) => opr.input(),
            Composer::Ternary(opr) => opr.input(),
            Composer::Variadic(opr) => opr.input(),
        }
    }

    pub fn output(&self) -> &TensorDesc {
        match self {
            Composer::Nullary(opr) => opr.output(),
            Composer::Unary(opr) => opr.output(),
            Composer::Binary(opr) => opr.output(),
            Composer::Ternary(opr) => opr.output(),
            Composer::Variadic(opr) => opr.output(),
        }
    }

    pub fn cat(&self) -> Category {
        match self {
            Composer::Nullary(opr) => opr.cat(),
            Composer::Unary(opr) => opr.cat(),
            Composer::Binary(opr) => opr.cat(),
            Composer::Ternary(opr) => opr.cat(),
            Composer::Variadic(opr) => opr.cat(),
        }
    }
}

impl Debug for Composer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Composer::Nullary(opr) => write!(f, "nullary.{:?}", opr),
            Composer::Unary(opr) => write!(f, "unary.{:?}", opr),
            Composer::Binary(opr) => write!(f, "binary.{:?}", opr),
            Composer::Ternary(opr) => write!(f, "ternary.{:?}", opr),
            Composer::Variadic(opr) => write!(f, "variadic.{:?}", opr),
        }
    }
}

pub struct Transform {
    opr: Composer,
    t_order: usize,
    x: SmallVec<[Fun; 3]>,
}

impl Transform {
    pub fn nullary<O>(opr: O) -> Self
    where
        O: Compose<0> + 'static,
    {
        Transform {
            opr: Composer::Nullary(Box::new(opr)),
            t_order: 0,
            x: smallvec![],
        }
    }

    pub fn unary<O>(opr: O, x: Fun) -> Self
    where
        O: Compose<1> + 'static,
    {
        Transform {
            opr: Composer::Unary(Box::new(opr)),
            t_order: x.t_order() + 1,
            x: smallvec![x],
        }
    }

    pub fn binary<O>(opr: O, x1: Fun, x2: Fun) -> Self
    where
        O: Compose<2> + 'static,
    {
        Transform {
            opr: Composer::Binary(Box::new(opr)),
            t_order: cmp::max(x1.t_order(), x2.t_order()) + 1,
            x: smallvec![x1, x2],
        }
    }

    pub fn ternary<O>(opr: O, x1: Fun, x2: Fun, x3: Fun) -> Self
    where
        O: Compose<3> + 'static,
    {
        Transform {
            opr: Composer::Ternary(Box::new(opr)),
            t_order: cmp::max(cmp::max(x1.t_order(), x2.t_order()), x3.t_order()) + 1,
            x: smallvec![x1, x2, x3],
        }
    }

    pub fn variadic<O>(opr: O, x: Vec<Fun>) -> Self
    where
        O: VariadicCompose + 'static,
    {
        let max_t_order = x
            .iter()
            .map(|a| a.t_order()) // each rank
            .max() // get max rank in parent gen
            .unwrap();

        let x = x.into_iter().collect();

        Transform {
            opr: Composer::Variadic(Box::new(opr)),
            t_order: max_t_order + 1,
            x,
        }
    }
}

impl Transform {
    pub fn t_order(&self) -> usize {
        self.t_order
    }

    pub fn input(&self) -> &[Fun] {
        &self.x
    }

    pub fn opr(&self) -> &Composer {
        &self.opr
    }

    pub fn grad(&self, y: &Fun, gy: &Fun) -> Vec<Option<Fun>> {
        match &self.opr {
            Composer::Nullary(_) => Vec::new(),
            Composer::Unary(opr) => opr.grad([&self.x[0]], y, gy).to_vec(),
            Composer::Binary(opr) => opr.grad([&self.x[0], &self.x[1]], y, gy).to_vec(),
            Composer::Ternary(opr) => opr
                .grad([&self.x[0], &self.x[1], &self.x[2]], y, gy)
                .to_vec(),
            Composer::Variadic(opr) => opr.grad(&self.x, y, gy),
        }
    }

    pub fn cat(&self) -> Category {
        self.opr.cat()
    }
}

impl Debug for Transform {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.opr {
            Composer::Nullary(opr) => write!(f, "{:?}", self.opr),
            Composer::Unary(opr) => write!(f, "{:?}", self.opr),
            Composer::Binary(opr) => write!(f, "{:?}", self.opr),
            Composer::Ternary(opr) => write!(f, "{:?}", self.opr),
            Composer::Variadic(opr) => write!(f, "{:?}", self.opr),
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
