use crate::shape::IndexIter;
use crate::tensor::data::Scalar;
use crate::tensor::Tensor;
use std::cell::Ref;

pub struct Iter<'a, T> {
    buffer: Ref<'a, [T]>,
    index_iter: IndexIter,
}

impl<'a, T> Iter<'a, T>
where
    T: Scalar,
{
    pub fn new(tensor: &'a Tensor) -> Self {
        Iter {
            buffer: tensor.buffer().unwrap(),
            index_iter: IndexIter::new(tensor.shape()),
        }
    }
}

impl<T> Iterator for Iter<'_, T>
where
    T: Scalar,
{
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.index_iter.next().map(|i| (i, self.buffer[i]))
    }
}

impl<T> DoubleEndedIterator for Iter<'_, T>
where
    T: Scalar,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.index_iter.next_back().map(|i| (i, self.buffer[i]))
    }
}

impl<T> ExactSizeIterator for Iter<'_, T>
where
    T: Scalar,
{
    fn len(&self) -> usize {
        self.index_iter.len()
    }
}

pub struct AlongAxisIter<'a> {
    t: &'a Tensor,
    axis: usize,
    index: usize,
}

impl<'a> AlongAxisIter<'a> {
    pub fn new(t: &'a Tensor, axis: usize) -> Self {
        AlongAxisIter { t, axis, index: 0 }
    }
}

impl<'a> Iterator for AlongAxisIter<'a> {
    type Item = Tensor;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.t.extents()[self.axis] {
            self.index += 1;
            Some(self.t.index(self.index - 1, self.axis))
        } else {
            None
        }
    }
}
