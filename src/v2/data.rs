use std::cell::RefCell;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::ops::Mul;
use std::rc::{Rc, Weak};

pub trait Scalar: Sized + Copy + Send + Sync + Debug + Display + 'static {}

impl<T> DataLiteral<T> for Vec<T>
    where
        T: Scalar,
{
    fn flat_iter<'a>(&'a self) -> Box<dyn Iterator<Item=T> + 'a> {
        Box::new(self.iter().cloned())
    }

    fn extents(&self) -> Vec<usize> {
        vec![self.len()]
    }
}

pub trait DataLiteral<T>
    where
        T: Scalar,
{
    fn flat_iter<'a>(&'a self) -> Box<dyn Iterator<Item=T> + 'a>;

    fn extents(&self) -> Vec<usize>;

    fn to_vec(&self) -> Vec<T> {
        self.flat_iter().collect()
    }
}

impl<T, E, const C: usize> DataLiteral<T> for [E; C]
    where
        E: DataLiteral<T>,
        T: Scalar,
{
    fn flat_iter<'a>(&'a self) -> Box<dyn Iterator<Item=T> + 'a> {
        Box::new(self.iter().flat_map(|a| a.flat_iter()))
    }

    fn extents(&self) -> Vec<usize> {
        let mut s = self[0].extents();
        s.insert(0, self.len());
        s
    }
}

impl<'a, E, T> DataLiteral<T> for &'a [E]
    where
        E: DataLiteral<T>,
        T: Scalar,
{
    fn flat_iter<'b>(&'b self) -> Box<dyn Iterator<Item=T> + 'b> {
        Box::new(self.iter().flat_map(|a| a.flat_iter()))
    }

    fn extents(&self) -> Vec<usize> {
        let mut s = self[0].extents();
        s.insert(0, self.len());
        s
    }
}

impl<T> DataLiteral<T> for T
    where
        T: Scalar,
{
    fn flat_iter(&self) -> Box<dyn Iterator<Item=T>> {
        Box::new(core::iter::once(*self))
    }

    fn extents(&self) -> Vec<usize> {
        Vec::new()
    }
}

impl Scalar for f32 {}

impl Scalar for i32 {}

impl Scalar for u32 {}
