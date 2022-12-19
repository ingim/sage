use crate::layers::Parameter;
use crate::ops;
use crate::ops::map::scalar;
use crate::ops::{Compose, Transform, VariadicCompose};
use crate::session::context::Context;
use crate::shape::Shape;
use crate::shape::{Axes, Axis, Extent};
use crate::tensor::data::{DataLiteral, DataType, Scalar};
use crate::tensor::{Tensor, TensorDesc};
use core::fmt;
use itertools::Itertools;
use std::borrow::{Borrow, BorrowMut};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::binary_heap::BinaryHeap;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Deref, DerefMut};
use std::rc::Rc;

pub enum Origin {
    Operation(Transform),
    Data(RefCell<Option<Tensor>>),
}

#[derive(Clone)]
pub struct Note {
    name: String,
}

#[derive(Clone)]
pub struct Fun {
    // Basic template for variable
    desc: TensorDesc,

    // defines var
    origin: Rc<Origin>,

    // user-side info to make life easier
    notes: Option<Note>,
}

// Differentiate variables, a.k.a. backpropagation

pub fn grad_param<'a, P>(y: &'a Fun, model: &'a P) -> HashMap<&'a Fun, Fun>
where
    P: Parameter + 'a,
{
    let mut params = Vec::new();
    model.params(&mut params);

    grad(y, params)
}

pub fn grad<'a, I>(y: &'a Fun, x: I) -> HashMap<&'a Fun, Fun>
where
    I: IntoIterator<Item = &'a Fun> + 'a,
{
    let mut queue = BinaryHeap::<Ranked<&Fun>>::new();
    let mut grads = HashMap::<&Fun, Fun>::new();

    // The 'genesis' gy/gy, (which always equals to 1)
    grads.insert(y, scalar(1.0, y.extents()));
    queue.push(Ranked::new(y, y.t_order()));

    while !queue.is_empty() {
        // must unwrap
        let y = queue.pop().unwrap().into_inner();
        let gy = grads.get(&y).unwrap();

        if let Some(op) = y.op() {
            let x = op.input();
            let gx = op.grad(y, gy);

            // insert (x, gx) pairs into grads hashmap
            for (x, gx) in x.iter().zip(gx.into_iter()) {
                // skip non-differentiable variables.
                if let Some(gx) = gx {
                    if gx.extents() != x.extents() {
                        println!("{:?}", op.cat());
                        panic!("grad shape error. check grad func def");
                    }

                    if !grads.contains_key(x) {
                        queue.push(Ranked::new(x, x.t_order()))
                    }
                    grads
                        .entry(x)
                        .and_modify(|v| *v = (&gx).add(&*v))
                        .or_insert_with(|| gx);
                }
            }
        }
    }

    let mut grads_retained = HashMap::new();
    for v in x {
        grads_retained.insert(v, grads.remove(v).unwrap());
    }

    grads_retained
}

/** Gradient checking using numerical gradient by finite differences.
 **/
pub fn grad_check(y: &Fun, x: &Fun, err: f32, ctx: &mut Context) -> bool {
    let gx_gt = x.grad(y).eval(ctx).to_host();

    let d = x.data().unwrap().to_host();
    let eps = 1e-4;

    let mut b = d.buffer::<f32>().unwrap().to_vec();
    println!("--------------");
    for ((i, val), (j, val_gt)) in d.iter::<f32>().zip(gx_gt.iter::<f32>()) {
        // + eps
        b[i] = val + eps;
        x.set_data(b.as_ref());
        x.device(ctx);
        ctx.data.clear();
        let y1 = y.copy().eval(ctx);

        // -eps
        b[i] = val - eps;
        x.set_data(b.as_ref());
        x.device(ctx);
        ctx.data.clear();
        let y2 = y.copy().eval(ctx);

        b[i] = val;

        let gx = ((&y1 - &y2).sum(&(0..y.rank()).collect_vec(), false) / (2.0 * eps))
            .eval(ctx)
            .to_host();

        println!("{} vs {}", val_gt, gx.buffer::<f32>().unwrap()[0]);

        if (val_gt - gx.buffer::<f32>().unwrap()[0]).abs() > err {
            return false;
        }
    }

    // back to original state
    x.set_data(b.as_ref());
    x.device(ctx);

    true
}

impl Fun {
    // ******************************** Constructors ******************************** //

    pub fn new<T>(data: T) -> Self
    where
        T: AsRef<Tensor>,
    {
        let data = data.as_ref().clone();

        Fun {
            desc: data.desc().clone(),
            origin: Rc::new(Origin::Data(RefCell::new(Some(data)))),
            notes: None,
        }
    }

    pub fn empty<E>(extents: E, data_type: DataType) -> Self
    where
        E: Extent,
    {
        Fun {
            desc: TensorDesc::new(extents, data_type),
            origin: Rc::new(Origin::Data(RefCell::new(None))),
            notes: None,
        }
    }

    pub fn from_nullary_op<O>(opr: O) -> Self
    where
        O: Compose<0> + 'static,
    {
        Fun {
            desc: opr.output().clone(),
            origin: Rc::new(Origin::Operation(Transform::nullary(opr))),
            notes: None,
        }
    }

    pub fn from_unary_op<O>(opr: O, x: Fun) -> Self
    where
        O: Compose<1> + 'static,
    {
        Fun {
            desc: opr.output().clone(),
            origin: Rc::new(Origin::Operation(Transform::unary(opr, x))),
            notes: None,
        }
    }

    pub fn from_binary_op<O>(opr: O, x1: Fun, x2: Fun) -> Self
    where
        O: Compose<2> + 'static,
    {
        Fun {
            desc: opr.output().clone(),
            origin: Rc::new(Origin::Operation(Transform::binary(opr, x1, x2))),
            notes: None,
        }
    }

    pub fn from_ternary_op<O>(opr: O, x1: Fun, x2: Fun, x3: Fun) -> Self
    where
        O: Compose<3> + 'static,
    {
        Fun {
            desc: opr.output().clone(),
            origin: Rc::new(Origin::Operation(Transform::ternary(opr, x1, x2, x3))),
            notes: None,
        }
    }

    pub fn from_variadic_op<O>(opr: O, x: Vec<Fun>) -> Self
    where
        O: VariadicCompose + 'static,
    {
        Fun {
            desc: opr.output().clone(),
            origin: Rc::new(Origin::Operation(Transform::variadic(opr, x))),
            notes: None,
        }
    }

    pub fn name<T>(self, name: T) -> Self
    where
        T: AsRef<str>,
    {
        let mut v = self;
        let name = name.as_ref().to_string();

        if v.notes.is_none() {
            v.notes = Some(Note { name })
        } else {
            v.notes.as_mut().unwrap().name = name;
        }
        v
    }

    // ******************************** Properties ******************************** //

    pub fn desc(&self) -> &TensorDesc {
        &self.desc
    }

    pub fn extents(&self) -> &[usize] {
        self.desc.extents()
    }

    pub fn extent<A>(&self, axis: A) -> usize
    where
        A: Axis,
    {
        self.desc.extent(axis)
    }

    pub fn shape(&self) -> &Shape {
        self.desc.shape()
    }

    pub fn rank(&self) -> usize {
        self.desc.rank()
    }

    pub fn size(&self) -> usize {
        self.desc.size()
    }

    pub fn data_type(&self) -> DataType {
        self.desc.data_type()
    }
    //
    // pub fn origin(&self) -> &Origin {
    //     self.origin.borrow()
    // }

    pub fn op(&self) -> Option<&Transform> {
        if let Origin::Operation(op) = self.origin.borrow() {
            return Some(op.borrow());
        }

        None
    }

    pub fn is_data(&self) -> bool {
        !self.op().is_some()
    }

    pub fn is_op(&self) -> bool {
        self.op().is_some()
    }

    // pub fn data(&self) -> Option<&Tensor> {
    //     match self.origin.borrow() {
    //         Origin::Data(t) => Some(t),
    //         _ => None
    //     }
    // }

    pub fn data(&self) -> Option<Tensor> {
        if let Origin::Data(t) = self.origin.borrow() {
            return RefCell::borrow(t).clone();
        }
        None
    }

    pub fn t_order(&self) -> usize {
        match self.op() {
            Some(op) => op.t_order(),
            None => 0,
        }
    }

    // pub(crate) fn to_weak(&self) -> WeakVar {
    //     WeakVar::from(self)
    // }
    // ******************************** grad ******************************** //

    pub fn grad<V>(&self, y: V) -> Fun
    where
        V: ToFun,
    {
        grad(&y.to_fun(), core::slice::from_ref(self))
            .into_values()
            .next()
            .unwrap()
    }

    // ******************************** Data ******************************** //

    pub fn eval(&self, ctx: &mut Context) -> Tensor {
        // data
        if self.is_data() {
            self.data().unwrap()
        }
        // op
        else if self.is_op() {
            ctx.eval([self]);
            ctx.data.get(self).unwrap().clone()
        }
        // empty
        else {
            panic!("cannot evaluate empty var");
        }
    }

    pub fn device(&self, ctx: &mut Context) {
        self.data().unwrap().device(ctx);
    }

    pub fn host(&self) {
        self.data().unwrap().host();
    }

    pub fn set_data<D, T>(&self, data: D)
    where
        D: DataLiteral<T>,
        T: Scalar,
    {
        self.data().unwrap().set_data(data)
    }

    pub fn set(&self, tensor: Tensor) {
        if let Origin::Data(data) = self.origin.borrow() {
            let mut data = RefCell::borrow_mut(data);
            *data.deref_mut() = Some(tensor);
        } else {
            panic!("can only assign tensor to the data variable");
        }
    }

    // ******************************** Core Utilities ******************************** //

    pub fn copy(&self) -> Fun {
        ops::map::copy(self)
    }

    pub fn abs(&self) -> Fun {
        ops::map::abs(self)
    }

    pub fn modular<V>(&self, x: V) -> Fun
    where
        V: ToFun,
    {
        ops::map::modular(self, x)
    }

    pub fn recip(&self) -> Fun {
        ops::map::recip(self)
    }

    pub fn log(&self) -> Fun {
        ops::map::log(self)
    }

    pub fn exp(&self) -> Fun {
        ops::map::exp(self)
    }

    pub fn square(&self) -> Fun {
        ops::map::square(self)
    }

    pub fn sqrt(&self) -> Fun {
        ops::map::sqrt(self)
    }

    pub fn pow<V>(&self, x: V) -> Fun
    where
        V: ToFun,
    {
        ops::map::pow(self, x)
    }

    pub fn sin(&self) -> Fun {
        ops::map::sin(self)
    }

    pub fn sinh(&self) -> Fun {
        ops::map::sinh(self)
    }

    pub fn cos(&self) -> Fun {
        ops::map::cos(self)
    }

    pub fn cosh(&self) -> Fun {
        ops::map::cosh(self)
    }

    pub fn tan(&self) -> Fun {
        ops::map::tan(self)
    }

    pub fn tanh(&self) -> Fun {
        ops::map::tanh(self)
    }

    pub fn asin(&self) -> Fun {
        ops::map::asin(self)
    }

    pub fn asinh(&self) -> Fun {
        ops::map::asinh(self)
    }

    pub fn acos(&self) -> Fun {
        ops::map::acos(self)
    }

    pub fn acosh(&self) -> Fun {
        ops::map::acosh(self)
    }

    pub fn atan(&self) -> Fun {
        ops::map::atan(self)
    }

    pub fn atanh(&self) -> Fun {
        ops::map::atanh(self)
    }

    pub fn erf(&self) -> Fun {
        ops::map::erf(self)
    }

    pub fn sign(&self) -> Fun {
        ops::map::sign(self)
    }

    pub fn minimum<V>(&self, x: V) -> Fun
    where
        V: ToFun,
    {
        ops::map::min(self, x)
    }

    pub fn maximum<V>(&self, x: V) -> Fun
    where
        V: ToFun,
    {
        ops::map::max(self, x)
    }

    pub fn ceil(&self) -> Fun {
        ops::map::ceil(self)
    }

    pub fn floor(&self) -> Fun {
        ops::map::floor(self)
    }

    pub fn round(&self) -> Fun {
        ops::map::round(self)
    }

    ///// shaping
    pub fn organize(&self) -> Fun {
        if self.shape().has_default_strides() {
            self.clone()
        } else {
            self.copy()
        }
    }

    pub fn int(&self) -> Fun {
        ops::map::int(self)
    }

    pub fn float(&self) -> Fun {
        ops::map::float(self)
    }

    pub fn reshape(&self, shape: Shape) -> Fun {
        ops::core::reshape(self, shape)
    }

    pub fn view<E>(&self, extent: E) -> Fun
    where
        E: Extent,
    {
        ops::core::view(self, extent)
    }

    pub fn expand<E>(&self, extent: E) -> Fun
    where
        E: Extent,
    {
        ops::core::expand(self, extent)
    }

    pub fn tr(&self) -> Fun {
        ops::core::tr(self)
    }

    pub fn transpose<A1, A2>(&self, axis1: A1, axis2: A2) -> Fun
    where
        A1: Axis,
        A2: Axis,
    {
        ops::core::transpose(self, axis1, axis2)
    }

    pub fn permute<A>(&self, axes: A) -> Fun
    where
        A: Axes,
    {
        ops::core::permute(self, axes)
    }

    pub fn squeeze<A>(&self, axis: A) -> Fun
    where
        A: Axis,
    {
        ops::core::squeeze(self, axis)
    }

    pub fn unsqueeze<A>(&self, axis: A) -> Fun
    where
        A: Axis,
    {
        ops::core::unsqueeze(self, axis)
    }

    pub fn index<A>(&self, axis: A, index: usize) -> Fun
    where
        A: Axis,
    {
        self.slice(axis, index, index + 1)
    }

    pub fn slice<A>(&self, axis: A, start: usize, end: usize) -> Fun
    where
        A: Axis,
    {
        ops::core::slice(self, axis, start, end)
    }

    pub fn unslice<V, A>(&self, axis: A, start: usize, end: usize) -> Fun
    where
        V: ToFun,
        A: Axis,
    {
        ops::core::unslice(self, axis, start, end)
    }

    pub fn gather<V, A>(&self, idx: V, axis: A) -> Fun
    where
        V: ToFun,
        A: Axis,
    {
        ops::core::gather(self, idx, axis)
    }

    pub fn scatter<V, A, E>(&self, idx: V, axis: A, extent: E) -> Fun
    where
        V: ToFun,
        A: Axis,
        E: Extent,
    {
        ops::core::scatter(self, idx, axis, extent)
    }

    //// GEMM

    pub fn matmul<V>(&self, x: V) -> Fun
    where
        V: ToFun,
    {
        ops::gemm::matmul(self, x)
    }

    pub fn matmul_batched<V>(&self, x: V) -> Fun
    where
        V: ToFun,
    {
        ops::gemm::matmul_batched(self, x)
    }

    ///// reduction
    pub fn sum<A>(&self, axes: A, preserve_axes: bool) -> Fun
    where
        A: Axes,
    {
        ops::reduce::sum(self, axes, preserve_axes)
    }

    pub fn mean<A>(&self, axes: A, preserve_axes: bool) -> Fun
    where
        A: Axes,
    {
        ops::reduce::mean(self, axes, preserve_axes)
    }

    pub fn var<A>(&self, axes: A, preserve_axes: bool) -> Fun
    where
        A: Axes,
    {
        ops::reduce::var(self, axes, preserve_axes)
    }

    pub fn prod<A>(&self, axes: A, preserve_axes: bool) -> Fun
    where
        A: Axes,
    {
        ops::reduce::prod(self, axes, preserve_axes)
    }

    pub fn max<A>(&self, axes: A, preserve_axes: bool) -> Fun
    where
        A: Axes,
    {
        ops::reduce::max(self, axes, preserve_axes)
    }

    pub fn min<A>(&self, axes: A, preserve_axes: bool) -> Fun
    where
        A: Axes,
    {
        ops::reduce::min(self, axes, preserve_axes)
    }

    pub fn argmax<A>(&self, axis: A, preserve_axes: bool) -> Fun
    where
        A: Axis,
    {
        ops::reduce::argmax(self, axis, preserve_axes)
    }

    pub fn argmin<A>(&self, axis: A, preserve_axes: bool) -> Fun
    where
        A: Axis,
    {
        ops::reduce::argmin(self, axis, preserve_axes)
    }
}

impl Eq for Fun {}

impl PartialEq for Fun {
    fn eq(&self, other: &Self) -> bool {
        let extents_eq = self.shape().eq(other.shape());
        let data_type_eq = self.data_type() == other.data_type();

        let origin_eq = Rc::ptr_eq(&self.origin, &other.origin);

        extents_eq && data_type_eq && origin_eq
    }
}

impl Hash for Fun {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.shape().strides().hash(state);

        Rc::as_ptr(&self.origin).hash(state);
    }
}

impl Debug for Fun {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let name = if let Some(note) = &self.notes {
            note.name.as_ref()
        } else {
            "unnamed"
        };

        let origin = match self.origin.borrow() {
            Origin::Operation(opr) => format!("{:?}", opr),
            Origin::Data(_) => "data".to_string(),
        };

        write!(f, "{}({}) -> {:?}", name, origin, &self.desc)
    }
}

// to prevent stack overflow from recursive drop (when the computational graph is dropped.)
// impl Drop for Var {
//     fn drop(&mut self) {
//         if let Some(op) = self.op() {
//             let mut queue = BinaryHeap::<Ranked<Var>>::new();
//
//             for v in op.input().iter().cloned() {
//                 queue.push(v.into_ranked());
//             }
//
//             while !queue.is_empty() {
//                 let var = queue.pop().unwrap().into_inner();
//
//
//                 if let Origin::Operation(op) = var.origin {
//                     if let Ok(op) = Rc::try_unwrap(op) {
//                         for v in op.input().iter().cloned() {
//                             queue.push(v.into_ranked());
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }
//
//
// impl WeakVar {
//     fn from(t: &Var) -> Self {
//         WeakVar {
//             shape: t.shape.clone(),
//             data_type: t.data_type,
//             origin: t.origin.as_ref().map(Rc::downgrade),
//         }
//     }
// }

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

// (kinda) special list for storing var refs
pub struct List<'a> {
    list: Vec<&'a Fun>,
}

impl<'a> List<'a> {
    pub fn new() -> Self {
        List { list: vec![] }
    }

    pub fn add(&mut self, var: &'a Fun) -> &mut Self {
        self.list.push(var);
        self
    }

    pub fn eval(self, ctx: &mut Context) {
        todo!()
    }
}

pub trait ToFun {
    fn to_fun(self) -> Fun;
}

impl ToFun for Fun {
    fn to_fun(self) -> Fun {
        self
    }
}

impl ToFun for &Fun {
    fn to_fun(self) -> Fun {
        (*self).clone()
    }
}
//
// impl Variable for WeakVar {
//     fn to_var(&self) -> Var {
//         Var {
//             shape: self.shape.clone(),
//             data_type: self.data_type,
//             origin: self.origin.as_ref().map(|o| o.upgrade().unwrap()),
//         }
//     }
// }

impl ToFun for Tensor {
    fn to_fun(self) -> Fun {
        Fun::new(self)
    }
}

impl ToFun for &Tensor {
    fn to_fun(self) -> Fun {
        Fun::new(self)
    }
}

impl<T> ToFun for T
where
    T: Scalar,
{
    fn to_fun(self) -> Fun {
        scalar(self, [1])
    }
}
