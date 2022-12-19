

pub trait Backend {

    type TensorPrimitive;
}


pub struct Tensor<B:Backend> {

    pub data: B::TensorPrimitive
}

