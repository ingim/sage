use crate::v2::function::Function;

pub trait Backend {
    type TensorPrimitive;

    fn default() -> Self::TensorPrimitive;
}


pub enum Tensor<B: Backend> {
    Lazy(Function<B>, Vec<Tensor<B>>),
    Ready(B::TensorPrimitive),
}

impl<B:Backend> Tensor<B> {
    pub fn new() -> Self {
        Tensor::Ready(B::default())
    }
}


pub struct Native {}


impl Backend for Native {
    type TensorPrimitive = String;

    fn default() -> Self::TensorPrimitive {
        "Hi. this is default value".to_string()
    }
}