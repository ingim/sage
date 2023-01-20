use std::borrow::Borrow;
use std::collections::HashMap;
use std::iter::zip;
use std::rc::Rc;
use crate::v2::backend::{Backend, TensorPrimitive};
use crate::v2::ir;
use crate::v2::shape::{Array, IndexIter, Shape};
use crate::v2::tensor::data::{Buffer, Scalar};

pub struct Native {}


impl Backend for Native {
    type Tensor = Tensor;

    fn eval(f: ir::Graph, inputs: HashMap<ir::Node, Self::Tensor>, outputs: Vec<ir::Node>) -> Vec<Self::Tensor> {

        // first optimize the graph
        let mut stack = outputs.clone();

        let mut visited = HashMap::<ir::Node, Self::Tensor>::new();
        let mut arg_buf = Vec::<Self::Tensor>::with_capacity(3);

        while let Some(node) = stack.pop() {
            let args = f.edges_in(node);

            let mut unvisited = args.iter().filter(|arg| !visited.contains_key(*arg)).peekable();

            if unvisited.peek().is_some() {
                stack.push(node);
                stack.extend(unvisited);
            } else {
                args.iter().map(|arg| visited[arg].clone()).collect_into(&mut arg_buf);

                let data = match node.cmd() {
                    ir::Command::Data => inputs[&node].clone(),
                    ir::Command::Full(scalar) => unimplemented!(),
                    ir::Command::Add => add(&arg_buf)
                };

                visited.insert(node, data);
                arg_buf.clear();
            }
        }

        outputs.into_iter().map(|node| visited[&node].clone()).collect()
    }
}


#[derive(Clone)]
pub struct Tensor {
    buffer: Rc<Buffer>,
    shape: Shape,
}

impl Tensor {
    pub fn new(buffer: Buffer, shape: Shape) -> Self {
        Tensor { buffer: Rc::new(buffer), shape }
    }

    pub fn extents(&self) -> &[usize] {
        self.shape.extents()
    }

    pub fn buffer<T: Scalar>(&self) -> &[T] {
        T::data_to_vec(&self.buffer)
    }

    pub fn iter<T: Scalar>(&self) -> Iter<T> {
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


pub fn add(args: &[Tensor]) -> Tensor {
    let data = match (args[0].buffer.borrow(), args[1].buffer.borrow()) {
        (Buffer::Float(a), Buffer::Float(b)) => binary_map(a, b, |a, b| a + b),
        (Buffer::Int(a), Buffer::Int(b)) => binary_map(a, b, |a, b| a + b),
        (Buffer::Uint(a), Buffer::Uint(b)) => binary_map(a, b, |a, b| a + b),
        _ => panic!("data type does not match")
    };

    Tensor::new(data, args[0].shape.clone())
}


pub fn binary_map<T: Scalar>(a: &[T], b: &[T], f: fn(T, T) -> T) -> Buffer
{
    T::vec_to_data(zip(a, b).map(|(a, b)| f(*a, *b)).collect())
}


pub struct Iter<'a, T>

{
    buffer: &'a [T],
    index_iter: IndexIter,
}

impl<'a, T> Iter<'a, T>
    where T: Scalar
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
