use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{Hash, Hasher};
use std::iter::zip;
use std::rc::Rc;
use crate::v2::ops::scalar;
use crate::v2::backend::{Backend};
use crate::v2::ir;
use crate::v2::utils::Ranked;


pub trait Operator<const N: usize, B: Backend> {
    // fn grad(&self, x: [&Fun; N], y: &Fun, gy: &Fun) -> [Option<Fun>; N];
    // fn compute(&self, x: [&Tensor; N], ctx: &mut Context) -> Result<Tensor, Error>;

    fn grad(&self, x: &[Tensor<B>; N], gy: &Tensor<B>) -> [Option<Tensor<B>>; N];

    fn build_ir(&self, x: &[ir::Node; N], g: &mut ir::Graph) -> ir::Node;
}

pub struct Operation<const N: usize, B: Backend> {
    f: Box<dyn Operator<N, B>>,
    args: [Tensor<B>; N],

    // Topological order of the function.
    t_order: usize,
}

impl<const N: usize, B: Backend> Operation<N, B> {
    pub fn new<O: Operator<N, B>>(f: O, args: [Tensor<B>; N]) -> Self {
        let t_order = args.iter().map(|x| x.t_order()).max().unwrap_or(0);
        Operation {
            f: Box::new(f),
            args,
            t_order: t_order + 1,
        }
    }

    pub fn t_order(&self) -> usize {
        self.t_order
    }
}

impl<B: Backend> Operation<0, B> {
    pub fn into_tensor(self) -> Tensor<B> {
        Prototype::Nullary(self).into_tensor()
    }
}

impl<B: Backend> Operation<1, B> {
    pub fn into_tensor(self) -> Tensor<B> {
        Prototype::Unary(self).into_tensor()
    }
}

impl<B: Backend> Operation<2, B> {
    pub fn into_tensor(self) -> Tensor<B> {
        Prototype::Binary(self).into_tensor()
    }
}

impl<B: Backend> Operation<3, B> {
    pub fn into_tensor(self) -> Tensor<B> {
        Prototype::Ternary(self).into_tensor()
    }
}


pub enum Prototype<B: Backend> {
    Constant,
    Nullary(Operation<0, B>),
    Unary(Operation<1, B>),
    Binary(Operation<2, B>),
    Ternary(Operation<3, B>),
}


impl<B: Backend> Prototype<B> {
    // Convert a prototype into a tensor.
    pub fn into_tensor(self) -> Tensor<B> {
        Tensor::from_prototype(self)
    }

    // Get the topological order of the prototype.
    pub fn t_order(&self) -> usize {
        match self {
            Prototype::Constant => 0,
            Prototype::Nullary(op) => 0,
            Prototype::Unary(op) => op.t_order(),
            Prototype::Binary(op) => op.t_order(),
            Prototype::Ternary(op) => op.t_order(),
        }
    }
}

#[derive(Clone)]
pub struct Tensor<B: Backend> {
    pub proto: Rc<Prototype<B>>,
    pub data: Rc<RefCell<Option<B::TensorPrimitive>>>,
}

impl<B: Backend> Tensor<B> {
    pub fn new() -> Self {
        Tensor { proto: Rc::new(Prototype::Constant) }
    }

    pub fn from_prototype(proto: Prototype<B>) -> Self {
        Tensor {
            proto: Rc::new(proto),
            data: Rc::new(RefCell::new(None)),
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        vec![0]
    }

    pub fn t_order(&self) -> usize {
        self.proto.t_order()
    }


    pub fn grad(&self, x: &Tensor<B>) -> Tensor<B> {
        Tensor::new()
    }


    pub fn sync(&self) {
        let mut data = self.data.borrow_mut();
        if data.is_none() {
            B::eval(self)

                * data = Some(B::new_tensor());
        }
    }
}


// grad!(f) is equivalent to |a, b| {
//      grad(f(a, b))
// }
// f


// trait Parameter { get_list() }


pub fn grad<'a, B: Backend>(y: Tensor<B>) -> HashMap<&'a Tensor<B>, Tensor<B>> {
    let mut queue = BinaryHeap::<Ranked<&Tensor<B>>>::new();
    let mut grads = HashMap::<&Tensor<B>, Tensor<B>>::new();

    // The 'genesis' gy/gy, (which always equals to 1)
    grads.insert(&y, scalar(1.0));
    queue.push(Ranked::new(&y, y.t_order()));

    while !queue.is_empty() {
        // must unwrap
        let y = queue.pop().unwrap().into_inner();
        let gy = grads.get(&y).unwrap();

        let (x, gx) = match y.proto.borrow() {
            Prototype::Unary(op) => {
                (op.args.as_slice(), op.f.grad(&op.args, gy).as_slice())
            }
            Prototype::Binary(op) => {
                (op.args.as_slice(), op.f.grad(&op.args, gy).as_slice())
            }
            Prototype::Ternary(op) => {
                (op.args.as_slice(), op.f.grad(&op.args, gy).as_slice())
            }
            // Nullary and Data are not differentiable.
            _ => {
                ([].as_slice(), [].as_slice())
            }
        };

        // insert (x, gx) pairs into grads hashmap
        for (x, gx) in zip(x, gx) {
            // skip non-differentiable variables.
            if let Some(gx) = gx {
                // if gx.shape() != x.shape() {
                //     panic!("grad shape error. check grad func def");
                // }

                if !grads.contains_key(x) {
                    queue.push(Ranked::new(x, x.t_order()))
                }

                grads
                    .entry(x)
                    .and_modify(|v| *v = (&gx).add(&*v))
                    .or_insert_with(|| *gx);
            }
        }
    }

    grads
}


pub fn sync<'a, B, I>(x: I)
    where B: Backend + 'a, I: IntoIterator<Item=&'a Tensor<B>> + 'a
{

    // sort x by t_order in descending order
    let mut x: Vec<&Tensor<B>> = x.into_iter().collect();
    x.sort_by(|a, b| b.t_order().cmp(&a.t_order()));


    // traverse x
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
        Rc::ptr_eq(&self.proto, &other.proto)
    }
}

impl<B: Backend> Hash for Tensor<B> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.proto).hash(state)
    }
}


/// Fn traits

impl<B: Backend> Fn<(i32, )> for Tensor<B> {
    extern "rust-call" fn call(&self, _args: (i32, )) {
        println!("One argument");
    }
}

impl<B: Backend> FnMut<(i32, )> for Tensor<B> {
    extern "rust-call" fn call_mut(&mut self, _args: (i32, )) {
        println!("One argument");
    }
}

impl<B: Backend> FnOnce<(i32, )> for Tensor<B> {
    type Output = ();

    extern "rust-call" fn call_once(self, _args: (i32, )) {
        println!("One argument");
    }
}


impl<B: Backend> Fn<(i32, i32)> for Tensor<B> {
    extern "rust-call" fn call(&self, _args: (i32, i32)) -> Self::Output {
        println!("Two argument");
        0
    }
}

impl<B: Backend> FnMut<(i32, i32)> for Tensor<B> {
    extern "rust-call" fn call_mut(&mut self, _args: (i32, i32)) -> Self::Output {
        println!("Two argument");
        0
    }
}

impl<B: Backend> FnOnce<(i32, i32)> for Tensor<B> {
    type Output = i32;

    extern "rust-call" fn call_once(self, _args: (i32, i32)) -> Self::Output {
        println!("Two argument");
        0
    }
}


