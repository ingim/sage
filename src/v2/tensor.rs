use std::borrow::{Borrow, BorrowMut};
use std::cell::RefCell;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::iter;
use std::iter::zip;
use std::ops::{Add, Deref};
use std::rc::Rc;
use itertools::Itertools;

use crate::v2::ops::scalar;
use crate::v2::backend::{Backend};
use crate::v2::ir;
use crate::v2::utils::Ranked;


pub trait Operator<const N: usize, B: Backend>: Clone {
    fn grad(&self, x: &[Tensor<B>; N], y: &Tensor<B>, gy: &Tensor<B>) -> [Option<Tensor<B>>; N];

    fn build_ir(&self, x: &[ir::Node; N], g: &mut ir::Graph) -> ir::Node;
}


struct Operation<B: Backend> {
    args: Vec<Tensor<B>>,
    t_order: usize,

    grad: Box<dyn Fn(&[Tensor<B>], &Tensor<B>, &Tensor<B>) -> Vec<Option<Tensor<B>>>>,
    build_ir: Box<dyn Fn(&[ir::Node], &mut ir::Graph) -> ir::Node>,
}


impl<B: Backend> Operation<B> {
    fn new<const N: usize, O>(op: O, args: [Tensor<B>; N]) -> Operation<B>
        where O: Operator<N, B> + 'static
    {
        let t_order = args.iter().map(|x| x.t_order()).max().unwrap_or(0);
        let op2 = op.clone();

        let grad = Box::new(move |x: &[Tensor<B>], y: &Tensor<B>, gy: &Tensor<B>| {
            match x.try_into() {
                Ok(x) => op.grad(x, y, gy),
                Err(_) => panic!("sds"),
            }.to_vec()
        });

        let build_ir = Box::new(move |x: &[ir::Node], g: &mut ir::Graph| {
            match x.try_into() {
                Ok(x) => op2.build_ir(x, g),
                Err(_) => panic!("sds"),
            }
        });

        Operation {
            grad,
            build_ir,
            args: args.to_vec(),
            t_order,
        }
    }

    pub fn args(&self) -> &[Tensor<B>] {
        &self.args
    }

    pub fn grad(&self, y: &Tensor<B>, gy: &Tensor<B>) -> Vec<Option<Tensor<B>>> {
        (self.grad)(&self.args, y, gy)
    }

    pub fn build_ir(&self, x: &[ir::Node], g: &mut ir::Graph) -> ir::Node {
        (self.build_ir)(x, g)
    }
}
//
//
// struct Op<const N: usize, B: Backend> {
//     f: Box<dyn Operator<N, B>>,
//     args: [Tensor<B>; N],
//
//     // Topological order of the function.
//     t_order: usize,
// }
//
// struct VariadicOp<B: Backend> {
//     f: Box<dyn VariadicOperator<B>>,
//     args: Vec<Tensor<B>>,
//
//     // Topological order of the function.
//     t_order: usize,
// }
//
// impl<const N: usize, B: Backend> Op<N, B> {
//     pub fn new<O>(f: O, args: [Tensor<B>; N]) -> Self
//         where O: Operator<N, B> + 'static
//     {
//         let t_order = args.iter().map(|x| x.t_order()).max().unwrap_or(0);
//         Op {
//             f: Box::new(f),
//             args,
//             t_order: t_order + 1,
//         }
//     }
// }
//
// impl<B: Backend> VariadicOp<B> {
//     pub fn new<O>(f: O, args: Vec<Tensor<B>>) -> Self
//         where O: VariadicOperator<B> + 'static
//     {
//         let t_order = args.iter().map(|x| x.t_order()).max().unwrap_or(0);
//         VariadicOp {
//             f: Box::new(f),
//             args,
//             t_order: t_order + 1,
//         }
//     }
// }

// enum Operation<B: Backend> {
//     Constant,
//     // Nullary(Op<0, B>),
//     // Unary(Op<1, B>),
//     // Binary(Op<2, B>),
//     // Ternary(Op<3, B>),
//     // Variadic(VariadicOp<B>),
// }


impl<B: Backend> Operation<B> {}

pub struct Tensor<B: Backend> {
    op: Rc<Option<Operation<B>>>,
    data: Rc<RefCell<Option<B::TensorPrimitive>>>,
}

impl<B: Backend> Tensor<B> {
    pub fn new() -> Self {
        Tensor { op: Rc::new(None), data: Rc::new(RefCell::new(None)) }
    }

    pub fn from_op<const N: usize, O>(op: O, args: [Tensor<B>; N]) -> Self
        where O: Operator<N, B> + 'static
    {
        let op = Operation::new(op, args);
        Tensor { op: Rc::new(Some(op)), data: Rc::new(RefCell::new(None)) }
    }

    //
    // pub fn from_nullary_op<O>(op: O) -> Self
    //     where O: Operator<0, B> + 'static
    // {
    //     let op = Operation::new(op, []);
    //     Tensor::from_op(Operation::Nullary(op))
    // }
    //
    // pub fn from_unary_op<O>(op: O, x: Tensor<B>) -> Self
    //     where O: Operator<1, B> + 'static
    // {
    //     let op = Operation::new(op, [x]);
    //     Tensor::from_op(Operation::Unary(op))
    // }
    //
    // pub fn from_binary_op<O>(op: O, x0: Tensor<B>, x1: Tensor<B>) -> Self
    //     where O: Operator<2, B> + 'static
    // {
    //     let op = Operation::new(op, [x0, x1]);
    //     Tensor::from_op(Operation::Binary(op))
    // }
    //
    // pub fn from_ternary_op<O>(op: O, x0: Tensor<B>, x1: Tensor<B>, x2: Tensor<B>) -> Self
    //     where O: Operator<3, B> + 'static
    // {
    //     let op = Operation::new(op, [x0, x1, x2]);
    //     Tensor::from_op(Operation::Ternary(op))
    // }
    //
    // pub fn from_variadic_op<O>(op: O, x: Vec<Tensor<B>>) -> Self
    //     where O: VariadicOperator<B> + 'static
    // {
    //     let op = VariadicOp::new(op, x);
    //     Tensor::from_op(Operation::Variadic(op))
    // }

    pub fn shape(&self) -> Vec<usize> {
        vec![0]
    }

    pub fn t_order(&self) -> usize {
        if let Some(op) = &*self.op {
            op.t_order
        } else {
            0
        }
    }


    pub fn grad(&self, x: &Tensor<B>) -> Tensor<B> {
        Tensor::new()
    }


    pub fn sync(&self) {}

    pub fn ready(&self) -> bool {
        RefCell::borrow(&self.data).is_some()
    }
}

impl<B: Backend> Clone for Tensor<B> {
    fn clone(&self) -> Self {
        Tensor {
            op: self.op.clone(),
            data: self.data.clone(),
        }
    }
}

pub fn grad<'a, B: Backend>(y: &'a Tensor<B>) -> HashMap<&'a Tensor<B>, Tensor<B>> {
    let mut queue = BinaryHeap::<Ranked<&Tensor<B>>>::new();
    let mut grads = HashMap::<&'a Tensor<B>, Tensor<B>>::new();
    // The 'genesis' gy/gy, (which always equals to 1)
    grads.insert(y, scalar(1.0));
    queue.push(Ranked::new(y, y.t_order()));

    while !queue.is_empty() {
        // must unwrap
        let y = queue.pop().unwrap().into_inner();
        let gy = grads.get(&y).unwrap();

        if let Some(op) = &*y.op {
            let x = op.args();
            let gx = op.grad(y, gy);

            for (x, gx) in zip(x, gx) {
                // skip non-differentiable variables.
                if let Some(gx) = gx {
                    if !grads.contains_key(x) {
                        queue.push(Ranked::new(x, x.t_order()))
                    }
                    grads
                        .entry(x)
                        .and_modify(|v| *v = gx.add(v))
                        .or_insert_with(|| gx);
                }
            }
        }
    }
    grads
}

pub fn sync<'a, B, I>(x: I)
    where B: Backend + 'a, I: IntoIterator<Item=&'a Tensor<B>> + 'a
{
    let mut g = ir::Graph::new();

    // sort x by t_order in descending order
    let mut x: Vec<&Tensor<B>> = x.into_iter().collect();
    x.sort_by_key(|b| std::cmp::Reverse(b.t_order()));

    let mut done = HashMap::<&Tensor<B>, ir::Node>::new();
    let mut node_args = Vec::with_capacity(3);

    // traverse x
    for x in x {

        // traverse until it meets a data node
        let mut stack = vec![x];

        while !stack.is_empty() {
            let e = stack.last().unwrap();

            // already exist in the graph
            if done.contains_key(e) {
                stack.pop();
                continue;
            }

            if e.ready() {
                let node = g.data();
                done.insert(e, node);
                stack.pop();
            }

            // attempt to insert into the graph
            else if let Some(op) = &*e.op {
                let mut not_done = op.args()
                    .iter()
                    .filter(|v| !done.contains_key(v))
                    .peekable();

                if not_done.peek().is_some() {
                    not_done.for_each(|v| stack.push(v));
                } else {
                    op.args().iter().map(|v| done[&v]).collect_into(&mut node_args);

                    let node = op.build_ir(&node_args, &mut g);
                    done.insert(e, node);
                    stack.pop();
                    node_args.clear();
                }
            } else {
                panic!("Uncomputable tensor");
            }
        }
    }
}


//
//
// //
// //
// //
// //
// // pub fn grad<'a, I>(y: &'a Fun, x: I) -> HashMap<&'a Fun, Fun>
// //     where
// //         I: IntoIterator<Item = &'a Fun> + 'a,
// // {
// //     let mut queue = BinaryHeap::<Ranked<&Fun>>::new();
// //     let mut grads = HashMap::<&Fun, Fun>::new();
// //
// //     // The 'genesis' gy/gy, (which always equals to 1)
// //     grads.insert(y, scalar(1.0, y.extents()));
// //     queue.push(Ranked::new(y, y.t_order()));
// //
// //     while !queue.is_empty() {
// //         // must unwrap
// //         let y = queue.pop().unwrap().into_inner();
// //         let gy = grads.get(&y).unwrap();
// //
// //         if let Some(op) = y.op() {
// //             let x = op.input();
// //             let gx = op.grad(y, gy);
// //
// //             // insert (x, gx) pairs into grads hashmap
// //             for (x, gx) in x.iter().zip(gx.into_iter()) {
// //                 // skip non-differentiable variables.
// //                 if let Some(gx) = gx {
// //                     if gx.extents() != x.extents() {
// //                         println!("{:?}", op.cat());
// //                         panic!("grad shape error. check grad func def");
// //                     }
// //
// //                     if !grads.contains_key(x) {
// //                         queue.push(Ranked::new(x, x.t_order()))
// //                     }
// //                     grads
// //                         .entry(x)
// //                         .and_modify(|v| *v = (&gx).add(&*v))
// //                         .or_insert_with(|| gx);
// //                 }
// //             }
// //         }
// //     }
// //
// //     let mut grads_retained = HashMap::new();
// //     for v in x {
// //         grads_retained.insert(v, grads.remove(v).unwrap());
// //     }
// //
// //     grads_retained
// // }
//
//
// impl<B: Backend> Eq for Function<B> {}
//
// impl<B: Backend> PartialEq for Function<B> {
//     fn eq(&self, other: &Self) -> bool {
//         Rc::ptr_eq(&self.op, &other.op)
//     }
// }
//
// impl<B: Backend> Hash for Function<B> {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         Rc::as_ptr(&self.op).hash(state);
//     }
// }


impl<B: Backend> Eq for Tensor<B> {}

impl<B: Backend> PartialEq for Tensor<B> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.op, &other.op)
    }
}

impl<B: Backend> Hash for Tensor<B> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.op).hash(state)
    }
}


impl<B: Backend> Add for &Tensor<B> {
    type Output = Tensor<B>;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor::new()
    }
}


//
// impl<B: Backend> Fn<(i32, )> for Tensor<B> {
//     extern "rust-call" fn call(&self, _args: (i32, )) {
//         println!("One argument");
//     }
// }
//
// impl<B: Backend> FnMut<(i32, )> for Tensor<B> {
//     extern "rust-call" fn call_mut(&mut self, _args: (i32, )) {
//         println!("One argument");
//     }
// }
//
// impl<B: Backend> FnOnce<(i32, )> for Tensor<B> {
//     type Output = ();
//
//     extern "rust-call" fn call_once(self, _args: (i32, )) {
//         println!("One argument");
//     }
// }
//
//
// impl<B: Backend> Fn<(i32, i32)> for Tensor<B> {
//     extern "rust-call" fn call(&self, _args: (i32, i32)) -> Self::Output {
//         println!("Two argument");
//         0
//     }
// }
//
// impl<B: Backend> FnMut<(i32, i32)> for Tensor<B> {
//     extern "rust-call" fn call_mut(&mut self, _args: (i32, i32)) -> Self::Output {
//         println!("Two argument");
//         0
//     }
// }
//
// impl<B: Backend> FnOnce<(i32, i32)> for Tensor<B> {
//     type Output = i32;
//
//     extern "rust-call" fn call_once(self, _args: (i32, i32)) -> Self::Output {
//         println!("Two argument");
//         0
//     }
// }
//

