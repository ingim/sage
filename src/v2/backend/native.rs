use std::borrow::Borrow;
use std::collections::HashMap;
use std::iter::zip;
use std::rc::Rc;
use crate::v2::backend::{Backend, TensorPrimitive};
use crate::v2::ir;
use crate::v2::shape::{Array, IndexIter, Shape};
use crate::v2::tensor::data::Scalar;

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
                    ir::Command::Data => inputs[&node].clone(),
                    ir::Command::Full(scalar, ..) => unimplemented!(),
                    ir::Command::Add => add(&arg_buf[0], &arg_buf[1]),
                };

                visited.insert(node, data);
                arg_buf.clear();
            }
        }

        g.targets().iter().map(|node| visited[&node].clone()).collect()
    }
}


enum Buffer {
    Float(Vec<f32>),
    Int(Vec<i32>),
}

trait BufferElement: Scalar {
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
    fn new(buffer: Buffer, shape: Shape) -> Self {
        Tensor { buffer: Rc::new(buffer), shape }
    }

    pub fn extents(&self) -> &[usize] {
        self.shape.extents()
    }

    fn buffer<T: BufferElement>(&self) -> &[T] {
        T::buffer_as_slice(&self.buffer)
    }

    fn iter<T: BufferElement>(&self) -> Iter<T> {
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
    Tensor::new(data, shape)
}


pub fn add(x0: &Tensor, x1: &Tensor) -> Tensor {
    let data = match (x0.buffer.borrow(), x1.buffer.borrow()) {
        (Buffer::Float(a), Buffer::Float(b)) => binary_map(a, b, |a, b| a + b),
        (Buffer::Int(a), Buffer::Int(b)) => binary_map(a, b, |a, b| a + b),
        _ => panic!("data type does not match")
    };

    Tensor::new(data, x0.shape.clone())
}


fn binary_map<T: BufferElement>(a: &[T], b: &[T], f: fn(T, T) -> T) -> Buffer
{
    T::vec_into_buffer(zip(a, b).map(|(a, b)| f(*a, *b)).collect())
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
            buffer: tensor.buffer(),
            index_iter: IndexIter::new(&tensor.shape()),
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
