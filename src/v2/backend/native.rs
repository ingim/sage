use std::borrow::Borrow;
use std::collections::HashMap;
use std::iter::zip;
use std::ops::Add;
use std::rc::Rc;
use itertools::Itertools;
use crate::v2::backend::{Backend, TensorPrimitive};
use crate::v2::ir;
use crate::v2::shape::{Array, Axis, IndexIter, Shape};
use crate::v2::data::{DataLiteral, Scalar};
use crate::v2::ir::{Command, UnaryOperation, BinaryOperation, TernaryOperation};

mod format;

pub struct Native {}


impl Backend for Native {
    type Tensor = Tensor;

    fn eval(g: ir::Graph, inputs: HashMap<ir::Node, Self::Tensor>) -> Vec<Self::Tensor> {

        // first optimize the graph
        let mut stack = g.targets().to_vec();

        let mut visited = HashMap::<ir::Node, Self::Tensor>::new();
        let mut arg_buf = Vec::<Self::Tensor>::with_capacity(3);

        while let Some(node) = stack.pop() {
            let args = g.edges_in(node);

            let mut unvisited = args.iter().filter(|arg| !visited.contains_key(*arg)).peekable();

            if unvisited.peek().is_some() {
                stack.push(node);
                stack.extend(unvisited);
            } else {
                args.iter().map(|arg| visited[arg].clone()).collect_into(&mut arg_buf);

                let data = match g.cmd(node) {
                    Command::Data => inputs[&node].clone(),
                    Command::Constant(scalar) => Tensor::new(*scalar),
                    Command::Map1(map_op) => map1(*map_op, &arg_buf[0]),
                    Command::Map2(map_op) => map2(*map_op, &arg_buf[0], &arg_buf[1]),
                    Command::Map3(map_op) => map3(*map_op, &arg_buf[0], &arg_buf[1], &arg_buf[2]),
                };

                visited.insert(node, data);
                arg_buf.clear();
            }
        }

        g.targets().iter().map(|node| visited[node].clone()).collect()
    }
}

pub enum Buffer {
    Float(Vec<f32>),
    Int(Vec<i32>),
}


impl Buffer {
    pub fn as_slice<T: BufferElement>(&self) -> &[T] {
        T::buffer_as_slice(self)
    }

    pub fn len(&self) -> usize {
        match self {
            Buffer::Float(v) => v.len(),
            Buffer::Int(v) => v.len(),
        }
    }
}


pub trait BufferElement: Scalar {
    fn vec_into_buffer(vec: Vec<Self>) -> Buffer;
    fn buffer_into_vec(buf: Buffer) -> Vec<Self>;
    fn buffer_as_slice(buf: &Buffer) -> &[Self];
}

impl BufferElement for f32 {
    fn vec_into_buffer(vec: Vec<Self>) -> Buffer {
        Buffer::Float(vec)
    }

    fn buffer_into_vec(buf: Buffer) -> Vec<Self> {
        match buf {
            Buffer::Float(v) => v,
            _ => panic!("data type does not match")
        }
    }

    fn buffer_as_slice(buf: &Buffer) -> &[Self] {
        match buf {
            Buffer::Float(v) => v,
            _ => panic!("data type does not match")
        }
    }
}


impl BufferElement for i32 {
    fn vec_into_buffer(vec: Vec<Self>) -> Buffer {
        Buffer::Int(vec)
    }

    fn buffer_into_vec(buf: Buffer) -> Vec<Self> {
        match buf {
            Buffer::Int(v) => v,
            _ => panic!("data type does not match")
        }
    }

    fn buffer_as_slice(buf: &Buffer) -> &[Self] {
        match buf {
            Buffer::Int(v) => v,
            _ => panic!("data type does not match")
        }
    }
}


#[derive(Clone)]
pub struct Tensor {
    buffer: Rc<Buffer>,
    shape: Shape,
}

impl Tensor {
    pub fn new<D, T>(data: D) -> Self
        where
            D: DataLiteral<T>,
            T: BufferElement,
    {
        let shape = Shape::new(data.extents());
        let buffer = BufferElement::vec_into_buffer(data.to_vec());
        Tensor { buffer: Rc::new(buffer), shape }
    }

    pub fn with_layout(&self, shape: Shape) -> Self {
        Tensor { buffer: self.buffer.clone(), shape }
    }

    pub fn extents(&self) -> &[usize] {
        self.shape.extents()
    }

    pub fn index<A>(&self, index: usize, axis: A) -> Tensor
        where
            A: Axis,
    {
        let axis = axis.to_usize(self.rank()).unwrap();
        let axis_size = self.extents()[axis];

        if axis_size <= index {
            panic!("index out of bounds");
        }

        self.with_layout(self.shape().select(index, axis))
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.num_axes()
    }

    pub fn size(&self) -> usize {
        self.shape.size()
    }

    pub fn buffer(&self) -> &Buffer {
        self.buffer.borrow()
    }

    pub fn iter<T: BufferElement>(&self) -> Iter<T> {
        Iter::new(self)
    }
}

impl TensorPrimitive for Tensor {
    fn shape(&self) -> Shape {
        self.shape.clone()
    }
}

pub fn full(shape: Shape, value: f32) -> Tensor {
    let data = Buffer::Float(vec![value; shape.size()]);
    Tensor { buffer: Rc::new(data), shape }
}

pub fn map1(map_op: UnaryOperation, x: &Tensor) -> Tensor {
    let data = match x.buffer.borrow() {
        Buffer::Float(a) => Buffer::Float(map1_f32(map_op, a)),
        Buffer::Int(a) => Buffer::Int(map1_i32(map_op, a)),
        _ => panic!("data type does not match")
    };
    Tensor { buffer: Rc::new(data), shape: x.shape.clone() }
}

fn map1_f32(map_op: UnaryOperation, a: &[f32]) -> Vec<f32>
{
    let f: fn(f32) -> f32 = match map_op {
        UnaryOperation::Id => |a| a,
        UnaryOperation::Abs => |a| a.abs(),
        UnaryOperation::Neg => |a| -a,
        UnaryOperation::Recip => |a| 1.0 / a,
        UnaryOperation::Log => |a| a.ln(),
        UnaryOperation::Exp => |a| a.exp(),
        UnaryOperation::Sqrt => |a| a.sqrt(),
        UnaryOperation::Square => |a| a * a,
        UnaryOperation::Sign => |a| a.signum(),
        UnaryOperation::Ceil => |a| a.ceil(),
        UnaryOperation::Floor => |a| a.floor(),
        UnaryOperation::Round => |a| a.round(),
        UnaryOperation::Sin => |a| a.sin(),
        UnaryOperation::Cos => |a| a.cos(),
        UnaryOperation::Tan => |a| a.tan(),
        UnaryOperation::Asin => |a| a.asin(),
        UnaryOperation::Acos => |a| a.acos(),
        UnaryOperation::Atan => |a| a.atan(),
        UnaryOperation::Sinh => |a| a.sinh(),
        UnaryOperation::Cosh => |a| a.cosh(),
        UnaryOperation::Tanh => |a| a.tanh(),
        UnaryOperation::Asinh => |a| a.asinh(),
        UnaryOperation::Acosh => |a| a.acosh(),
        UnaryOperation::Atanh => |a| a.atanh(),
        UnaryOperation::IsNan => |a| a.is_nan() as i32 as f32,
        UnaryOperation::IsInf => |a| a.is_infinite() as i32 as f32,
        UnaryOperation::Not => |a| if a > 0.0 { 0.0 } else { 1.0 },
        _ => panic!("not implemented")
    };
    a.iter().map(|&a| f(a)).collect()
}

fn map1_i32(map_op: UnaryOperation, a: &[i32]) -> Vec<i32>
{
    let f: fn(i32) -> i32 = match map_op {
        UnaryOperation::Id => |a| a,
        UnaryOperation::Abs => |a| a.abs(),
        UnaryOperation::Neg => |a| -a,
        UnaryOperation::Square => |a| a * a,
        UnaryOperation::Sign => |a| a.signum(),
        UnaryOperation::Not => |a| if a > 0 { 0 } else { 1 },
        _ => panic!("not implemented")
    };
    a.iter().map(|&a| f(a)).collect()
}


pub fn map2(map_op: BinaryOperation, x0: &Tensor, x1: &Tensor) -> Tensor {
    let data = match (x0.buffer.borrow(), x1.buffer.borrow()) {
        (Buffer::Float(a), Buffer::Float(b)) => Buffer::Float(map2_f32(map_op, a, b)),
        (Buffer::Int(a), Buffer::Int(b)) => Buffer::Int(map2_i32(map_op, a, b)),
        _ => panic!("data type does not match")
    };
    Tensor { buffer: Rc::new(data), shape: x0.shape.clone() }
}


fn map2_f32(map_op: BinaryOperation, a: &[f32], b: &[f32]) -> Vec<f32>
{
    let f: fn(f32, f32) -> f32 = match map_op {
        BinaryOperation::Add => |a, b| a + b,
        BinaryOperation::Sub => |a, b| a - b,
        BinaryOperation::Div => |a, b| a / b,
        BinaryOperation::Mul => |a, b| a * b,
        BinaryOperation::Mod => |a, b| a % b,
        BinaryOperation::Pow => |a, b| a.powf(b),
        BinaryOperation::Min => |a, b| a.min(b),
        BinaryOperation::Max => |a, b| a.max(b),
        BinaryOperation::And => |a, b| if a > 0.0 && b > 0.0 { 1.0 } else { 0.0 },
        BinaryOperation::Or => |a, b| if a > 0.0 || b > 0.0 { 1.0 } else { 0.0 },
        BinaryOperation::Eq => |a, b| if a == b { 1.0 } else { 0.0 },
        BinaryOperation::Ne => |a, b| if a != b { 1.0 } else { 0.0 },
        BinaryOperation::Gt => |a, b| if a > b { 1.0 } else { 0.0 },
        BinaryOperation::Ge => |a, b| if a >= b { 1.0 } else { 0.0 },
        BinaryOperation::Lt => |a, b| if a < b { 1.0 } else { 0.0 },
        BinaryOperation::Le => |a, b| if a <= b { 1.0 } else { 0.0 },
    };

    zip(a, b).map(|(a, b)| f(*a, *b)).collect()
}


fn map2_i32(map_op: BinaryOperation, a: &[i32], b: &[i32]) -> Vec<i32>
{
    let f: fn(i32, i32) -> i32 = match map_op {
        BinaryOperation::Add => |a, b| a + b,
        BinaryOperation::Sub => |a, b| a - b,
        BinaryOperation::Div => |a, b| a / b,
        BinaryOperation::Mul => |a, b| a * b,
        BinaryOperation::Mod => |a, b| a % b,
        BinaryOperation::Pow => |a, b| a.pow(b as u32),
        BinaryOperation::Min => |a, b| a.min(b),
        BinaryOperation::Max => |a, b| a.max(b),
        BinaryOperation::And => |a, b| if a > 0 && b > 0 { 1 } else { 0 },
        BinaryOperation::Or => |a, b| if a > 0 || b > 0 { 1 } else { 0 },
        BinaryOperation::Eq => |a, b| if a == b { 1 } else { 0 },
        BinaryOperation::Ne => |a, b| if a != b { 1 } else { 0 },
        BinaryOperation::Gt => |a, b| if a > b { 1 } else { 0 },
        BinaryOperation::Ge => |a, b| if a >= b { 1 } else { 0 },
        BinaryOperation::Lt => |a, b| if a < b { 1 } else { 0 },
        BinaryOperation::Le => |a, b| if a <= b { 1 } else { 0 },
    };

    zip(a, b).map(|(a, b)| f(*a, *b)).collect()
}

pub fn map3(map_op: TernaryOperation, x0: &Tensor, x1: &Tensor, x2: &Tensor) -> Tensor {
    let data = match (x0.buffer.borrow(), x1.buffer.borrow(), x2.buffer.borrow()) {
        (Buffer::Float(a), Buffer::Float(b), Buffer::Float(c)) => Buffer::Float(map3_f32(map_op, a, b, c)),
        (Buffer::Int(a), Buffer::Int(b), Buffer::Int(c)) => Buffer::Int(map3_i32(map_op, a, b, c)),
        _ => panic!("data type does not match")
    };
    Tensor { buffer: Rc::new(data), shape: x0.shape.clone() }
}

fn map3_f32(map_op: TernaryOperation, a: &[f32], b: &[f32], c: &[f32]) -> Vec<f32>
{
    let f: fn(f32, f32, f32) -> f32 = match map_op {
        TernaryOperation::Cond => |a, b, c| if a > 0.0 { b } else { c }
    };

    zip(a, zip(b, c)).map(|(a, (b, c))| f(*a, *b, *c)).collect()
}

fn map3_i32(map_op: TernaryOperation, a: &[i32], b: &[i32], c: &[i32]) -> Vec<i32>
{
    let f: fn(i32, i32, i32) -> i32 = match map_op {
        TernaryOperation::Cond => |a, b, c| if a > 0 { b } else { c }
    };

    zip(a, zip(b, c)).map(|(a, (b, c))| f(*a, *b, *c)).collect()
}


pub struct Iter<'a, T>

{
    buffer: &'a [T],
    index_iter: IndexIter,
}

impl<'a, T> Iter<'a, T>
    where T: BufferElement
{
    pub fn new(tensor: &'a Tensor) -> Self {
        Iter {
            buffer: tensor.buffer().as_slice(),
            index_iter: IndexIter::new(tensor.shape()),
        }
    }
}

impl<T> Iterator for Iter<'_, T>
    where T: Scalar

{
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.index_iter.next().map(|i| (i, self.buffer[i]))
    }
}

impl<T> DoubleEndedIterator for Iter<'_, T>
    where T: Scalar

{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.index_iter.next_back().map(|i| (i, self.buffer[i]))
    }
}

impl<T> ExactSizeIterator for Iter<'_, T>
    where T: Scalar

{
    fn len(&self) -> usize {
        self.index_iter.len()
    }
}
//
// pub struct AlongAxisIter<'a> {
//     t: &'a Tensor,
//     axis: usize,
//     index: usize,
// }
//
// impl<'a> AlongAxisIter<'a> {
//     pub fn new(t: &'a Tensor, axis: usize) -> Self {
//         AlongAxisIter { t, axis, index: 0 }
//     }
// }
//
// impl<'a> Iterator for AlongAxisIter<'a> {
//     type Item = Tensor;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         if self.index < self.t.extents()[self.axis] {
//             self.index += 1;
//             Some(self.t.index(self.index - 1, self.axis))
//         } else {
//             None
//         }
//     }
// }
//

// TODO today - write a iterator that visits tensor buffer based on the shape layout.
