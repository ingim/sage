#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![allow(unused)]  // FIXME
#![feature(is_sorted)]
#![feature(iter_collect_into)]

extern crate core;

mod v2;

// local create sage-macros
use sage_macros::differentiable;
use crate::v2::tensor::Tensor;
use crate::v2::backend::{Backend};
use crate::v2::backend::native::Native;
use v2::data::Scalar;

struct Param {
    x: Tensor,
}


fn main() {


    // condition 1. the native form must be usable.


    // grad(some_calc)()

    // grad(some_calc) -> Function

    // data().iter()

    // let y = SomeOp.some_calc(x);
//

    let a = Tensor::new(3.0);

    println!("{:?}", &a);

    a.data().iter().for_each(|x: (usize, f32)| println!("sd"));
}


//
//
// fn grad(f) -> Function {
//
//     // make sure f is differentiable
//
//     // first invoke a function
//     // 1. f(Dummy1, Dummy2, Dummy3,...) -> (Lazy) Tensor
//     // extract the function from the tensor
//
//
// }


/*

macro fn jit(f) -> Function







 */

