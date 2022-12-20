#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![allow(unused)]  // FIXME

mod v2;

// local create sage-macros
use sage_macros::differentiable;
use v2::tensor::Tensor;
use crate::v2::tensor::{Backend, Native};

struct SomeOp;

impl SomeOp {

    // tensors are lazy
    // enum Tensor {
    //     Lazy(OtherTensors, Function),
    //     Ready(Tensor),
    // }

    // Example:
    fn some_calc<B: Backend>(&self, x1: Tensor<B>, x2: Tensor<B>) -> Tensor<B> {

        // rules
        // 1. No &mut self
        // 2. No &mut x
        // 3. must return single tensor




        Tensor { data: B::default() }
    }
// out: attr: "delimiters"
// out: item: "fn invoke4() {}"
}




fn main() {
    let x = Tensor::<Native> { data: Native::default() };

    // condition 1. the native form must be usable.


    // grad(some_calc)()

    // grad(some_calc) -> Function

    // let y = SomeOp.some_calc(x);
//




    println!("{:?}", a);
    println!("{:?}", b);


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

