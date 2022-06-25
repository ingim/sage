pub mod mnist;

pub trait Dataset {
    type Item;
    fn is_empty(&self) -> bool;
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Option<Self::Item>;
    fn iter(&self) -> Iter<Self::Item>;
}

pub struct Iter<'a, T> {
    dataset: &'a dyn Dataset<Item = T>,
    index: usize,
}

impl<'a, T> Iter<'a, T> {
    pub fn new(dataset: &'a impl Dataset<Item = T>) -> Self {
        Iter { dataset, index: 0 }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index;

        if self.dataset.len() > index {
            self.index += 1;
            self.dataset.get(index)
        } else {
            None
        }
    }
}

impl<T> Iter<'_, T> {
    pub fn batch<F>(self, size: usize, f: F) -> Batch<Self, F> {
        Batch {
            iter: self,
            size,
            skip_last: true,
            f,
        }
    }
}

pub struct Batch<I, F> {
    iter: I,
    size: usize,
    skip_last: bool,
    f: F,
}

impl<I, F, B> Iterator for Batch<I, F>
where
    I: Iterator,
    F: Fn(&[I::Item]) -> Option<B>,
{
    type Item = B;

    fn next(&mut self) -> Option<B> {
        let mut batch: Vec<I::Item> = Vec::with_capacity(self.size);

        for _ in 0..self.size {
            if let Some(item) = self.iter.next() {
                batch.push(item);
            }
        }

        if !batch.is_empty() && (!self.skip_last || batch.len() == self.size) {
            (self.f)(&batch)
        } else {
            None
        }
    }
}
