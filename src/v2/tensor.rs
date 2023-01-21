use std::borrow::{Borrow, BorrowMut};
use std::cell::RefCell;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::{fmt, iter};
use std::iter::zip;
use std::marker::PhantomData;
use std::ops::{Add, Deref};
use std::rc::Rc;
use itertools::Itertools;

use crate::v2::ops::scalar;
use crate::v2::backend::{Backend, TensorPrimitive};
use crate::v2::backend::native::Native;
use crate::v2::{backend, ir};
use crate::v2::shape::Shape;
use crate::v2::data::Scalar;
use crate::v2::data::DataLiteral;
use crate::v2::utils::Ranked;


pub trait Operator<const N: usize, B: Backend>: Clone {
    fn grad(&self, x: &[Tensor<B>; N], y: &Tensor<B>, gy: &Tensor<B>) -> [Option<Tensor<B>>; N];

    fn build_ir(&self, x: [ir::Node; N], g: &mut ir::Graph) -> ir::Node;
}

struct Operation<B: Backend> {
    args: Vec<Tensor<B>>,
    t_order: usize,

    grad: Box<dyn Fn(&[Tensor<B>], &Tensor<B>, &Tensor<B>) -> Vec<Option<Tensor<B>>>>,
    build_ir: Box<dyn Fn(&[ir::Node], &mut ir::Graph) -> ir::Node>,
}

impl<B: Backend> Operation<B> {
    fn new<const N: usize, O>(op: O, args: [Tensor<B>; N]) -> Self
        where O: Operator<N, B> + 'static
    {
        let t_order = args.iter().map(|x| x.t_order()).max().unwrap_or(0);
        let op2 = op.clone();

        let grad = Box::new(move |x: &[Tensor<B>], y: &Tensor<B>, gy: &Tensor<B>| {
            match x.try_into() {
                Ok(x) => op.grad(x, y, gy),
                Err(_) => panic!("this should never happen"),
            }.to_vec()
        });

        let build_ir = Box::new(move |x: &[ir::Node], g: &mut ir::Graph| {
            match x.try_into() {
                Ok(x) => op2.build_ir(x, g),
                Err(_) => panic!("this should never happen"),
            }
        });

        Operation {
            grad,
            build_ir,
            args: args.to_vec(),
            t_order,
        }
    }

    pub fn args(&self) -> &[Tensor<B>] {
        &self.args
    }

    pub fn grad(&self, y: &Tensor<B>, gy: &Tensor<B>) -> Vec<Option<Tensor<B>>> {
        (self.grad)(&self.args, y, gy)
    }

    pub fn build_ir(&self, x: &[ir::Node], g: &mut ir::Graph) -> ir::Node {
        (self.build_ir)(x, g)
    }
}


pub struct Tensor<B: Backend = Native> {
    op: Rc<Option<Operation<B>>>,
    data: Rc<RefCell<Option<B::Tensor>>>,
}

impl<B: Backend> Tensor<B> {
    pub fn from_op<const N: usize, O>(op: O, args: [Tensor<B>; N]) -> Self
        where O: Operator<N, B> + 'static
    {
        let op = Operation::new(op, args);
        Tensor { op: Rc::new(Some(op)), data: Rc::new(RefCell::new(None)) }
    }

    pub fn shape(&self) -> Shape {
        self.data().shape()
    }

    pub fn t_order(&self) -> usize {
        if let Some(op) = &*self.op {
            op.t_order
        } else {
            0
        }
    }

    pub fn grad(&self, x: &Tensor<B>) -> Tensor<B> {
        todo!()
    }

    fn ready(&self) -> bool {
        RefCell::borrow(&self.data).is_some()
    }

    pub fn data(&self) -> B::Tensor {
        if !self.ready() {
            eval([self]);
        }
        // must unwrap because we just checked that it is ready
        RefCell::borrow(&self.data).as_ref().cloned().unwrap()
    }

    // impl operations




}

// native only
impl Tensor {
    pub fn new<D, T>(data: D) -> Self
        where
            D: DataLiteral<T>,
            T: backend::native::BufferElement,
    {
        let data = backend::native::Tensor::new(data);

        Tensor {
            op: Rc::new(None),
            data: Rc::new(RefCell::new(Some(data))),
        }
    }
}

impl<B: Backend> Tensor<B> {}


pub fn grad<'a, B: Backend>(y: &'a Tensor<B>) -> HashMap<&'a Tensor<B>, Tensor<B>> {
    let mut queue = BinaryHeap::<Ranked<&Tensor<B>>>::new();
    let mut grads = HashMap::<&'a Tensor<B>, Tensor<B>>::new();
    // The 'genesis' gy/gy, (which always equals to 1)
    grads.insert(y, scalar(1.0));
    queue.push(Ranked::new(y, y.t_order()));

    while !queue.is_empty() {
        // must unwrap
        let y = queue.pop().unwrap().into_inner();
        let gy = grads.get(&y).unwrap();

        if let Some(op) = &*y.op {
            let x = op.args();
            let gx = op.grad(y, gy);

            for (x, gx) in zip(x, gx) {
                // skip non-differentiable variables.
                if let Some(gx) = gx {
                    if !grads.contains_key(x) {
                        queue.push(Ranked::new(x, x.t_order()))
                    }
                    grads
                        .entry(x)
                        .and_modify(|v| *v = gx.add(v))
                        .or_insert_with(|| gx);
                }
            }
        }
    }
    grads
}

pub fn eval<'a, B, I>(x: I)
    where B: Backend + 'a, I: IntoIterator<Item=&'a Tensor<B>> + 'a
{
    let mut targets: Vec<&Tensor<B>> = x.into_iter().collect();
    // sort x by t_order in descending order
    targets.sort_by_key(|b| std::cmp::Reverse(b.t_order()));

    let mut g = ir::Graph::new();
    let mut inputs = HashMap::<ir::Node, B::Tensor>::new();

    let mut done = HashMap::<&Tensor<B>, ir::Node>::new();
    let mut node_args = Vec::with_capacity(3);

    // traverse x
    for &x in targets.iter() {

        // traverse until it meets a data node
        let mut stack = vec![x];

        while !stack.is_empty() {
            let e = stack.last().unwrap();

            // already exist in the graph
            if done.contains_key(e) {
                stack.pop();
                continue;
            }

            if let Some(data) = RefCell::borrow(&e.data).as_ref() {
                let node = g.data();
                inputs.insert(node, data.clone());

                done.insert(e, node);
                stack.pop();
            }

            // attempt to insert into the graph
            else if let Some(op) = &*e.op {
                let mut not_done = op.args()
                    .iter()
                    .filter(|v| !done.contains_key(v))
                    .peekable();

                if not_done.peek().is_some() {
                    stack.extend(not_done);
                } else {
                    op.args().iter().map(|v| done[&v]).collect_into(&mut node_args);

                    let node = op.build_ir(&node_args, &mut g);
                    done.insert(e, node);
                    stack.pop();
                    node_args.clear();
                }
            } else {
                panic!("Uncomputable tensor");
            }
        }
    }

    targets.iter().for_each(|v| g.add_target(done[v]));

    let res = B::eval(g, inputs);

    for (target, r) in zip(targets, res) {
        RefCell::borrow_mut(&target.data).replace(r);
    }
}


impl<B: Backend> Clone for Tensor<B> {
    fn clone(&self) -> Self {
        Tensor {
            op: self.op.clone(),
            data: self.data.clone(),
        }
    }
}

impl<B: Backend> Eq for Tensor<B> {}

impl<B: Backend> PartialEq for Tensor<B> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.op, &other.op)
    }
}

impl<B: Backend> Hash for Tensor<B> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.op).hash(state)
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.data())
    }
}


impl<B: Backend> Add for &Tensor<B> {
    type Output = Tensor<B>;

    fn add(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

