use std::cmp::Ordering;
use crate::v2::tensor::Tensor;
use crate::v2::backend::Backend;

pub struct Ranked<T> {
    inner: T,
    rank: usize,
}

impl<T> Ranked<T> {
    pub fn new(inner: T, rank: usize) -> Self {
        Ranked { inner, rank }
    }

    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<T> Eq for Ranked<T> {}

impl<T> PartialEq for Ranked<T> {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

impl<T> Ord for Ranked<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank.cmp(&other.rank)
    }
}

impl<T> PartialOrd for Ranked<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}


pub trait Parameter<B: Backend> {
    fn to_vec(&self) -> Vec<&Tensor<B>>;
}

