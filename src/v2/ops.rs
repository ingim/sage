use crate::v2::tensor::{Operation, Operator, Tensor};
use crate::v2::backend::Backend;
use std::marker::PhantomData;
use crate::v2::ir::{Graph, Node};
use crate::v2::shape::{Extent, Shape};

pub struct Full {
    scalar: f32,
    shape: Shape,
}


pub fn scalar<B: Backend>(scalar: f32) -> Tensor<B> {
    full(scalar, 1)
}


pub fn full<B: Backend, E: Extent>(scalar: f32, extent: E) -> Tensor<B> {
    Operation::<0, B>::new(Full { scalar, shape: Shape::new(extent) }, []).into_tensor()
}


impl<B: Backend> Operator<0, B> for Full {
    fn grad(&self, x: &[Tensor<B>; 0], gy: &Tensor<B>) -> [Option<Tensor<B>>; 0] {
        todo!()
    }

    fn build_ir(&self) -> String {
        "sd".to_string()
    }
}


pub struct Add;

impl<B: Backend> Operator<2, B> for Add {
    fn grad(&self, x: &[Tensor<B>; 2], gy: &Tensor<B>) -> [Option<Tensor<B>>; 2] {
        todo!()
    }

    fn build_ir(&self, x: &[Node; 2], g: &mut Graph) -> Node {
        let a = g.add(x[0], x[1]);
        g.add(a, x[1])
    }
}