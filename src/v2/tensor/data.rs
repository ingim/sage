use std::cell::RefCell;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::ops::Mul;
use std::rc::Weak;

// OpenCL data types
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataType {
    // Integers
    Int,
    // Unsigned Integers
    Uint,
    // Floats
    Float,
}

// buffer to store host-side data
#[derive(Clone, Debug)]
pub enum Buffer {
    Int(Vec<i32>),
    Uint(Vec<u32>),
    Float(Vec<f32>),
}


impl Display for DataType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            DataType::Int => write!(f, "int"),
            DataType::Uint => write!(f, "unsigned int"),
            DataType::Float => write!(f, "float"),
        }
    }
}

impl DataType {
    pub fn bytes(&self) -> usize {
        match self {
            DataType::Int => 4,
            DataType::Uint => 4,
            DataType::Float => 4,
        }
    }
}

pub trait Scalar: Sized + Copy + Send + Sync + Debug + Display + 'static {
    fn data_type() -> DataType;
    fn vec_to_data(v: Vec<Self>) -> Buffer;
    fn data_to_vec(a: &Buffer) -> &[Self];
}

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

    fn to_buf(&self) -> Buffer {
        T::vec_to_data(self.to_vec())
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

impl Scalar for f32 {
    fn data_type() -> DataType {
        DataType::Float
    }

    fn vec_to_data(v: Vec<f32>) -> Buffer {
        Buffer::Float(v)
    }

    fn data_to_vec(a: &Buffer) -> &[Self] {
        if let Buffer::Float(v) = a {
            v
        } else {
            panic!("not convertable");
        }
    }
}

impl Scalar for i32 {
    fn data_type() -> DataType {
        DataType::Int
    }

    fn vec_to_data(v: Vec<i32>) -> Buffer {
        Buffer::Int(v)
    }

    fn data_to_vec(a: &Buffer) -> &[Self] {
        if let Buffer::Int(v) = a {
            v
        } else {
            panic!("not convertable");
        }
    }
}

impl Scalar for u32 {
    fn data_type() -> DataType {
        DataType::Uint
    }

    fn vec_to_data(v: Vec<u32>) -> Buffer {
        Buffer::Uint(v)
    }

    fn data_to_vec(a: &Buffer) -> &[Self] {
        if let Buffer::Uint(v) = a {
            v
        } else {
            panic!("not convertable");
        }
    }
}
