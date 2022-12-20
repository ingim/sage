use std::collections::BinaryHeap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use crate::v2::tensor::{Backend, Tensor};
use crate::v2::utils::Ranked;


pub trait Compose<const N: usize, B: Backend> {
    // fn grad(&self, x: [&Fun; N], y: &Fun, gy: &Fun) -> [Option<Fun>; N];
    // fn compute(&self, x: [&Tensor; N], ctx: &mut Context) -> Result<Tensor, Error>;

    fn forward(&self, x: [Function<B>; N]) -> Function<B>;
    fn backward(&self, x: [Function<B>; N], gy: Function<B>) -> Function<B>;

    fn ir(&self) -> String;
}

pub struct Composition<const N: usize, B: Backend> {
    f: Box<dyn Compose<N, B>>,
    args: [Function<B>; N],

    // Topological order of the function.
    t_order: usize,
}


pub enum Operation<B: Backend> {
    Nullary(Composition<0, B>),
    Unary(Composition<1, B>),
    Binary(Composition<2, B>),
    Ternary(Composition<3, B>),
    Data(Option<Tensor<B>>), // Data(RefCell<Option<Tensor>>),??
}

pub struct Function<B: Backend> {
    pub op: Rc<Operation<B>>,
}
//
// pub fn grad<B: Backend>(f: Function<B>) -> Function<B> {
//     let mut queue = BinaryHeap::<Ranked<&Function>>::new();
//     let mut grads = HashMap::<&Fun, Fun>::new();
//
//     // The 'genesis' gy/gy, (which always equals to 1)
//     grads.insert(y, scalar(1.0, y.extents()));
//     queue.push(Ranked::new(y, y.t_order()));
//
//     while !queue.is_empty() {
//         // must unwrap
//         let y = queue.pop().unwrap().into_inner();
//         let gy = grads.get(&y).unwrap();
//
//         if let Some(op) = y.op() {
//             let x = op.input();
//             let gx = op.grad(y, gy);
//
//             // insert (x, gx) pairs into grads hashmap
//             for (x, gx) in x.iter().zip(gx.into_iter()) {
//                 // skip non-differentiable variables.
//                 if let Some(gx) = gx {
//                     if gx.extents() != x.extents() {
//                         println!("{:?}", op.cat());
//                         panic!("grad shape error. check grad func def");
//                     }
//
//                     if !grads.contains_key(x) {
//                         queue.push(Ranked::new(x, x.t_order()))
//                     }
//                     grads
//                         .entry(x)
//                         .and_modify(|v| *v = (&gx).add(&*v))
//                         .or_insert_with(|| gx);
//                 }
//             }
//         }
//     }
//
//     let mut grads_retained = HashMap::new();
//     for v in x {
//         grads_retained.insert(v, grads.remove(v).unwrap());
//     }
//
//     grads_retained
// }
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




struct Foo {}





impl Fn<(i32,)> for Foo {
    extern "rust-call" fn call(&self, _args: (i32,)) {
        println!("One argument");
    }
}

impl FnMut<(i32,)> for Foo {
    extern "rust-call" fn call_mut(&mut self, _args: (i32,)) {
        println!("One argument");
    }
}

impl FnOnce<(i32,)> for Foo {
    type Output = ();

    extern "rust-call" fn call_once(self, _args: (i32,)) {
        println!("One argument");
    }
}


impl Fn<(i32, i32)> for Foo {
    extern "rust-call" fn call(&self, _args: (i32,i32)) -> Self::Output {
        println!("Two argument");
        return 0;
    }
}

impl FnMut<(i32,i32)> for Foo {
    extern "rust-call" fn call_mut(&mut self, _args: (i32,i32)) -> Self::Output {
        println!("Two argument");
        return 0;
    }
}

impl FnOnce<(i32,i32)> for Foo {
    type Output = i32;

    extern "rust-call" fn call_once(self, _args: (i32,i32)) -> Self::Output {
        println!("Two argument");
        return 0;
    }
}


