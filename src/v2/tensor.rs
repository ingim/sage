pub trait Backend {
    type TensorPrimitive;

    fn default() -> Self::TensorPrimitive;
}


pub struct Tensor<B: Backend> {
    pub data: B::TensorPrimitive,
}


pub struct Native {}


impl Backend for Native {
    type TensorPrimitive = String;

    fn default() -> Self::TensorPrimitive {
        "Hi. this is default value".to_string()
    }
}