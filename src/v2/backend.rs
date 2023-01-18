use crate::v2::ir;
use crate::v2::tensor::Tensor;

pub trait Backend {
    type TensorPrimitive;

    fn default() -> Self::TensorPrimitive;

    fn eval(x: ir::Graph) -> Self::TensorPrimitive;
}


pub struct Native {}


impl Backend for Native {
    type TensorPrimitive = String;

    fn default() -> Self::TensorPrimitive {
        "Hi. this is default value".to_string()
    }
}