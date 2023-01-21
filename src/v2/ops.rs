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



pub fn map1<B: Backend>(op: UnaryOperation, x: Tensor<B>) -> Tensor<B> {
    Tensor::from_op(Map1 { op }, [x])
}


pub fn map2<B: Backend>(op: BinaryOperation, x0: Tensor<B>, x1: Tensor<B>) -> Tensor<B> {
    Tensor::from_op(Map2 { op }, [x0, x1])
}

pub fn map3<B: Backend>(op: TernaryOperation, x0: Tensor<B>, x1: Tensor<B>, x2: Tensor<B>) -> Tensor<B> {
    Tensor::from_op(Map3 { op }, [x0, x1, x2])
}


impl<B: Backend> Operator<1, B> for Map1 {
    fn grad(&self, x: &[Tensor<B>; 1], y: &Tensor<B>, gy: &Tensor<B>) -> [Option<Tensor<B>>; 1] {
        let x = &x[0];
        [match self.op {
            UnaryOperation::Id => Some(gy.clone()),
            UnaryOperation::Abs => Some(sign(x) * gy),
            UnaryOperation::Neg => Some(-gy.clone()),
            UnaryOperation::Recip => Some(-gy / x.square()),
            UnaryOperation::Log => Some(gy / x),
            UnaryOperation::Exp => Some(gy * y),
            UnaryOperation::Sqrt => Some(gy / (y * 2.0)),
            UnaryOperation::Square => Some(gy * x * 2.0),
            UnaryOperation::Sign => Some(scalar(0.0)),
            UnaryOperation::Ceil => Some(scalar(0.0)),
            UnaryOperation::Floor => Some(scalar(0.0)),
            UnaryOperation::Round => Some(scalar(0.0)),
            UnaryOperation::Sin => Some(gy * x.cos()),
            UnaryOperation::Sinh => Some(gy * x.cosh()),
            UnaryOperation::Cos => Some(-gy * x.sin()),
            UnaryOperation::Cosh => Some(gy * x.sinh()),
            UnaryOperation::Tan => Some(gy / x.cos().square()),
            UnaryOperation::Tanh => Some(gy / x.cosh().square()),
            UnaryOperation::Asin => Some(gy / (-x.square() + 1.0).sqrt()),
            UnaryOperation::Asinh => Some(gy / (x.square() + 1.0).sqrt()),
            UnaryOperation::Acos => Some(-gy / (-x.square() + 1.0).sqrt()),
            UnaryOperation::Acosh => Some(gy / (x.square() - 1.0).sqrt()),
            UnaryOperation::Atan => Some(gy / (x.square() + 1.0)),
            UnaryOperation::Atanh => Some(gy / (-x.square() + 1.0)),
            _ => None
        }]
    }

    fn build_ir(&self, x: [Node; 1], g: &mut Graph) -> Node {
        g.map1(self.op, x[0])
    }
}

