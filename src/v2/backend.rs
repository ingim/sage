pub mod native;

use std::collections::HashMap;
use crate::v2::ir;
use crate::v2::ir::{Command, Graph, NodeId};
use crate::v2::shape::Shape;
use crate::v2::data::Scalar;
use crate::v2::tensor::Tensor;

pub trait TensorPrimitive
{
    fn shape(&self) -> Shape;
}


pub trait Backend
{
    type Tensor: Clone + TensorPrimitive ;

    fn eval(f: ir::Graph, inputs: HashMap<ir::NodeId, Self::Tensor>) -> Vec<Self::Tensor>;
}
