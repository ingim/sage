use crate::v2::tensor::{Operator, Tensor};
use crate::v2::backend::Backend;
use std::marker::PhantomData;
use smallvec::{SmallVec, smallvec, ToSmallVec};
use crate::v2::ir::{BinaryOperation, Graph, Node, TernaryOperation, UnaryOperation};
use crate::v2::shape::{Extent, Shape};

#[derive(Clone)]
pub struct Full {
    scalar: f32,
    shape: Shape,
}

#[derive(Clone)]
pub struct Map1 {
    op: UnaryOperation,
}

#[derive(Clone)]
pub struct Map2 {
    op: BinaryOperation,
}

#[derive(Clone)]
pub struct Map3 {
    op: TernaryOperation,
}

pub fn scalar<B: Backend>(scalar: f32) -> Tensor<B> {
    full(scalar, 1)
}


pub fn full<B: Backend, E: Extent>(scalar: f32, extent: E) -> Tensor<B> {
    Tensor::from_op(Full { scalar, shape: Shape::new(extent) }, [])
}


impl<B: Backend> Operator<0, B> for Full {
    fn grad(&self, x: &[Tensor<B>; 0], _: &Tensor<B>, gy: &Tensor<B>) -> [Option<Tensor<B>>; 0] {
        todo!()
    }

    fn build_ir(&self, x: [Node; 0], g: &mut Graph) -> Node {
        g.constant(self.scalar)
    }
}

#[derive(Clone)]
pub struct Add;

impl<B: Backend> Operator<2, B> for Add {
    fn grad(&self, x: &[Tensor<B>; 2], _: &Tensor<B>, gy: &Tensor<B>) -> [Option<Tensor<B>>; 2] {
        todo!()
    }

    fn build_ir(&self, x: [Node; 2], g: &mut Graph) -> Node {
        g.add(x[0], x[1])
    }
}

