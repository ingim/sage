
use crate::v2::tensor::{IntoTensor, Operator, Tensor};
use crate::v2::backend::Backend;
use std::marker::PhantomData;
use std::ops;
use smallvec::{SmallVec, smallvec, ToSmallVec};
use crate::v2::data::DataLiteral;
use crate::v2::ir::{BinaryOperation, Graph, NodeId, TernaryOperation, UnaryOperation};
use crate::v2::shape;
use crate::v2::shape::{Extent, Shape};


impl<B: Backend> Tensor<B> {

    pub fn copy(&self) -> Tensor<B> {
        copy(self)
    }

    pub fn abs(&self) -> Tensor<B> {
        abs(self)
    }
    pub fn recip(&self) -> Tensor<B> {
        recip(self)
    }
    pub fn log(&self) -> Tensor<B> {
        log(self)
    }
    pub fn exp(&self) -> Tensor<B> {
        exp(self)
    }
    pub fn sqrt(&self) -> Tensor<B> {
        sqrt(self)
    }
    pub fn square(&self) -> Tensor<B> {
        square(self)
    }
    pub fn sign(&self) -> Tensor<B> {
        sign(self)
    }
    pub fn ceil(&self) -> Tensor<B> {
        ceil(self)
    }
    pub fn floor(&self) -> Tensor<B> {
        floor(self)
    }
    pub fn round(&self) -> Tensor<B> {
        round(self)
    }
    pub fn sin(&self) -> Tensor<B> {
        sin(self)
    }
    pub fn sinh(&self) -> Tensor<B> {
        sinh(self)
    }
    pub fn cos(&self) -> Tensor<B> {
        cos(self)
    }
    pub fn cosh(&self) -> Tensor<B> {
        cosh(self)
    }
    pub fn tan(&self) -> Tensor<B> {
        tan(self)
    }
    pub fn tanh(&self) -> Tensor<B> {
        tanh(self)
    }
    pub fn asin(&self) -> Tensor<B> {
        asin(self)
    }
    pub fn asinh(&self) -> Tensor<B> {
        asinh(self)
    }
    pub fn acos(&self) -> Tensor<B> {
        acos(self)
    }
    pub fn acosh(&self) -> Tensor<B> {
        acosh(self)
    }
    pub fn atan(&self) -> Tensor<B> {
        atan(self)
    }
    pub fn atanh(&self) -> Tensor<B> {
        atanh(self)
    }

    pub fn modulo<T: IntoTensor<B>>(&self, x: T) -> Tensor<B> {
        modulo(self, x)
    }


    pub fn pow<T: IntoTensor<B>>(&self, x: T) -> Tensor<B> {
        pow(self, x)
    }


    pub fn min<T: IntoTensor<B>>(&self, x: T) -> Tensor<B> {
        min(self, x)
    }

    pub fn max<T: IntoTensor<B>>(&self, x: T) -> Tensor<B> {
        max(self, x)
    }


    pub fn and<T: IntoTensor<B>>(&self, x: T) -> Tensor<B> {
        and(self, x)
    }


    pub fn or<T: IntoTensor<B>>(&self, x: T) -> Tensor<B> {
        or(self, x)
    }


    pub fn equal<T: IntoTensor<B>>(&self, x: T) -> Tensor<B> {
        eq(self, x)
    }


    pub fn ne<T: IntoTensor<B>>(&self, x: T) -> Tensor<B> {
        ne(self, x)
    }


    pub fn gt<T: IntoTensor<B>>(&self, x: T) -> Tensor<B> {
        gt(self, x)
    }


    pub fn ge<T: IntoTensor<B>>(&self, x: T) -> Tensor<B> {
        ge(self, x)
    }


    pub fn lt<T: IntoTensor<B>>(&self, x: T) -> Tensor<B> {
        lt(self, x)
    }


    pub fn le<T: IntoTensor<B>>(&self, x: T) -> Tensor<B> {
        le(self, x)
    }

    pub fn cond<T1: IntoTensor<B>, T2: IntoTensor<B>>(&self, x1: T1, x2: T2) -> Tensor<B> {
        cond(self, x1, x2)
    }
}


pub fn broadcast<B, T1, T2>(x1: T1, x2: T2) -> (Tensor, Tensor)
    where
        B: Backend,
        T1: IntoTensor<B>,
        T2: IntoTensor<B>,

{
    let mut x1 = x1.into_tensor();
    let mut x2 = x2.into_tensor();

    let union = shape::union(x1.extents(), x2.extents()).unwrap();

    if x1.extents() != union.as_slice() {
        x1 = x1.expand(&union);
    }
    if x2.extents() != union.as_slice() {
        x2 = x2.expand(&union);
    }
    (x1, x2)
}

pub fn broadcast3<B, T1, T2, T3>(x1: T1, x2: T2, x3: T3) -> (Tensor, Tensor, Tensor)
    where
        B: Backend,
        T1: IntoTensor<B>,
        T2: IntoTensor<B>,
        T3: IntoTensor<B>,
{
    let mut x1 = x1.into_tensor();
    let mut x2 = x2.into_tensor();
    let mut x3 = x3.into_tensor();

    let union1 = shape::union(x1.extents(), x2.extents()).unwrap();
    let union2 = shape::union(x2.extents(), x3.extents()).unwrap();
    let union = shape::union(&union1, &union2).unwrap();

    if x1.extents() != union.as_slice() {
        x1 = x1.expand(&union);
    }
    if x2.extents() != union.as_slice() {
        x2 = x2.expand(&union);
    }

    if x3.extents() != union.as_slice() {
        x3 = x3.expand(&union);
    }
    (x1, x2, x3)
}


#[derive(Clone)]
pub struct Full {
    scalar: f32,
    shape: Shape,
}

#[derive(Clone)]
pub struct Map1 {
    op: UnaryOperation,
}

#[derive(Clone)]
pub struct Map2 {
    op: BinaryOperation,
}

#[derive(Clone)]
pub struct Map3 {
    op: TernaryOperation,
}

pub fn scalar<B: Backend>(scalar: f32) -> Tensor<B> {
    full(scalar, 1)
}


pub fn full<B: Backend, E: Extent>(scalar: f32, extent: E) -> Tensor<B> {
    Tensor::from_op(Full { scalar, shape: Shape::new(extent) }, [])
}


impl<B: Backend> Operator<0, B> for Full {
    fn grad(&self, x: &[Tensor<B>; 0], _: &Tensor<B>, gy: &Tensor<B>) -> [Option<Tensor<B>>; 0] {
        todo!()
    }

    fn build(&self, x: [NodeId; 0], g: &mut Graph) -> NodeId {
        g.constant(self.scalar)
    }
}


pub fn map1<B: Backend>(op: UnaryOperation, x: Tensor<B>) -> Tensor<B> {
    Tensor::from_op(Map1 { op }, [x])
}

pub fn map2<B: Backend>(op: BinaryOperation, x0: Tensor<B>, x1: Tensor<B>) -> Tensor<B> {
    Tensor::from_op(Map2 { op }, [x0, x1])
}

pub fn map3<B: Backend>(op: TernaryOperation, x0: Tensor<B>, x1: Tensor<B>, x2: Tensor<B>) -> Tensor<B> {
    Tensor::from_op(Map3 { op }, [x0, x1, x2])
}

impl<B: Backend> Operator<1, B> for Map1 {
    fn grad(&self, x: &[Tensor<B>; 1], y: &Tensor<B>, gy: &Tensor<B>) -> [Option<Tensor<B>>; 1] {
        let x = &x[0];
        [match self.op {
            UnaryOperation::Copy => Some(gy.clone()),
            UnaryOperation::Abs => Some(sign(x) * gy),
            UnaryOperation::Neg => Some(-gy.clone()),
            UnaryOperation::Recip => Some(-gy / x.square()),
            UnaryOperation::Log => Some(gy / x),
            UnaryOperation::Exp => Some(gy * y),
            UnaryOperation::Sqrt => Some(gy / (y * 2.0)),
            UnaryOperation::Square => Some(gy * x * 2.0),
            UnaryOperation::Sign => Some(scalar(0.0)),
            UnaryOperation::Ceil => Some(scalar(0.0)),
            UnaryOperation::Floor => Some(scalar(0.0)),
            UnaryOperation::Round => Some(scalar(0.0)),
            UnaryOperation::Sin => Some(gy * x.cos()),
            UnaryOperation::Sinh => Some(gy * x.cosh()),
            UnaryOperation::Cos => Some(-gy * x.sin()),
            UnaryOperation::Cosh => Some(gy * x.sinh()),
            UnaryOperation::Tan => Some(gy / x.cos().square()),
            UnaryOperation::Tanh => Some(gy / x.cosh().square()),
            UnaryOperation::Asin => Some(gy / (-x.square() + 1.0).sqrt()),
            UnaryOperation::Asinh => Some(gy / (x.square() + 1.0).sqrt()),
            UnaryOperation::Acos => Some(-gy / (-x.square() + 1.0).sqrt()),
            UnaryOperation::Acosh => Some(gy / (x.square() - 1.0).sqrt()),
            UnaryOperation::Atan => Some(gy / (x.square() + 1.0)),
            UnaryOperation::Atanh => Some(gy / (-x.square() + 1.0)),
            _ => None
        }]
    }

    fn build(&self, x: [NodeId; 1], g: &mut Graph) -> NodeId {
        g.map1(self.op, x[0])
    }
}


impl<B: Backend> Operator<2, B> for Map2 {
    fn grad(&self, x: &[Tensor<B>; 2], _: &Tensor<B>, gy: &Tensor<B>) -> [Option<Tensor<B>>; 2] {
        let x1 = &x[0];
        let x2 = &x[1];

        match self.op {
            BinaryOperation::Add => [Some(gy.clone()), Some(gy.clone())],
            BinaryOperation::Sub => [Some(gy.clone()), Some(-gy.clone())],
            BinaryOperation::Div => [Some(gy / x2), Some(gy * (-x1 / x2.square()))],
            BinaryOperation::Mul => [Some(gy * x2), Some(gy * x1)],
            BinaryOperation::Mod => [Some(gy.clone()), Some(-gy * (x1 / x2).floor())],
            BinaryOperation::Pow => [
                Some(gy * x2 * x1.pow(x2 - 1.0)),
                Some(gy * x1.pow(x2) * x1.log()),
            ],
            BinaryOperation::Min => [
                Some(gy * lt(x1, x2)),
                Some(gy * ge(x1, x2)),
            ],
            BinaryOperation::Max => [
                Some(gy * lt(x2, x1)),
                Some(gy * ge(x2, x1)),
            ],
            BinaryOperation::And => [None, None],
            BinaryOperation::Or => [None, None],
            BinaryOperation::Eq => [None, None],
            BinaryOperation::Ne => [None, None],
            BinaryOperation::Gt => [None, None],
            BinaryOperation::Ge => [None, None],
            BinaryOperation::Lt => [None, None],
            BinaryOperation::Le => [None, None],
            _ => [None, None],
        }
    }

    fn build(&self, x: [NodeId; 2], g: &mut Graph) -> NodeId {
        g.map2(self.op, x[0], x[1])
    }
}


impl<B: Backend> Operator<3, B> for Map3 {
    fn grad(&self, x: &[Tensor<B>; 3], y: &Tensor<B>, gy: &Tensor<B>) -> [Option<Tensor<B>>; 3] {
        let x1 = &x[0];
        let x2 = &x[1];
        let x3 = &x[2];

        match self.op {
            TernaryOperation::Cond => [Some(cond(x1, gy, 0.0)), Some(cond(x1, 0.0, gy)), None],
            _ => [None, None, None],
        }
    }

    fn build(&self, x: [NodeId; 3], g: &mut Graph) -> NodeId {
        g.map3(self.op, x[0], x[1], x[2])
    }
}

pub fn copy<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Copy, x.into_tensor())
}


pub fn abs<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Abs, x.into_tensor())
}


pub fn neg<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Neg, x.into_tensor())
}


pub fn recip<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Recip, x.into_tensor())
}


pub fn log<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Log, x.into_tensor())
}


pub fn exp<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Exp, x.into_tensor())
}


pub fn sqrt<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Sqrt, x.into_tensor())
}

pub fn square<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Square, x.into_tensor())
}


pub fn sign<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Sign, x.into_tensor())
}


pub fn ceil<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Ceil, x.into_tensor())
}


pub fn floor<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Floor, x.into_tensor())
}

pub fn round<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Round, x.into_tensor())
}


pub fn sin<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Sin, x.into_tensor())
}

pub fn sinh<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Sinh, x.into_tensor())
}

pub fn cos<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Cos, x.into_tensor())
}

pub fn cosh<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Cosh, x.into_tensor())
}

pub fn tan<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Tan, x.into_tensor())
}


pub fn tanh<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Tanh, x.into_tensor())
}


pub fn asin<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Asin, x.into_tensor())
}

pub fn asinh<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Asinh, x.into_tensor())
}

pub fn acos<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Acos, x.into_tensor())
}

pub fn acosh<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Acosh, x.into_tensor())
}

pub fn atan<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Atan, x.into_tensor())
}

pub fn atanh<B, T>(x: T) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    map1(UnaryOperation::Atanh, x.into_tensor())
}


pub fn add<B, T1, T2>(x1: T1, x2: T2) -> Tensor<B>
    where B: Backend, T1: IntoTensor<B>, T2: IntoTensor<B>
{
    map2(BinaryOperation::Add, x1.into_tensor(), x2.into_tensor())
}

pub fn sub<B, T1, T2>(x1: T1, x2: T2) -> Tensor<B>
    where B: Backend, T1: IntoTensor<B>, T2: IntoTensor<B>
{
    map2(BinaryOperation::Sub, x1.into_tensor(), x2.into_tensor())
}

pub fn div<B, T1, T2>(x1: T1, x2: T2) -> Tensor<B>
    where B: Backend, T1: IntoTensor<B>, T2: IntoTensor<B>
{
    map2(BinaryOperation::Div, x1.into_tensor(), x2.into_tensor())
}


pub fn mul<B, T1, T2>(x1: T1, x2: T2) -> Tensor<B>
    where B: Backend, T1: IntoTensor<B>, T2: IntoTensor<B>
{
    map2(BinaryOperation::Mul, x1.into_tensor(), x2.into_tensor())
}


pub fn modulo<B, T1, T2>(x1: T1, x2: T2) -> Tensor<B>
    where B: Backend, T1: IntoTensor<B>, T2: IntoTensor<B>
{
    map2(BinaryOperation::Mod, x1.into_tensor(), x2.into_tensor())
}


pub fn pow<B, T1, T2>(x1: T1, x2: T2) -> Tensor<B>
    where B: Backend, T1: IntoTensor<B>, T2: IntoTensor<B>
{
    map2(BinaryOperation::Pow, x1.into_tensor(), x2.into_tensor())
}


pub fn min<B, T1, T2>(x1: T1, x2: T2) -> Tensor<B>
    where B: Backend, T1: IntoTensor<B>, T2: IntoTensor<B>
{
    map2(BinaryOperation::Min, x1.into_tensor(), x2.into_tensor())
}


pub fn max<B, T1, T2>(x1: T1, x2: T2) -> Tensor<B>
    where B: Backend, T1: IntoTensor<B>, T2: IntoTensor<B>
{
    map2(BinaryOperation::Max, x1.into_tensor(), x2.into_tensor())
}


pub fn and<B, T1, T2>(x1: T1, x2: T2) -> Tensor<B>
    where B: Backend, T1: IntoTensor<B>, T2: IntoTensor<B>
{
    map2(BinaryOperation::And, x1.into_tensor(), x2.into_tensor())
}


pub fn or<B, T1, T2>(x1: T1, x2: T2) -> Tensor<B>
    where B: Backend, T1: IntoTensor<B>, T2: IntoTensor<B>
{
    map2(BinaryOperation::Or, x1.into_tensor(), x2.into_tensor())
}


pub fn eq<B, T1, T2>(x1: T1, x2: T2) -> Tensor<B>
    where B: Backend, T1: IntoTensor<B>, T2: IntoTensor<B>
{
    map2(BinaryOperation::Eq, x1.into_tensor(), x2.into_tensor())
}


pub fn ne<B, T1, T2>(x1: T1, x2: T2) -> Tensor<B>
    where B: Backend, T1: IntoTensor<B>, T2: IntoTensor<B>
{
    map2(BinaryOperation::Ne, x1.into_tensor(), x2.into_tensor())
}


pub fn gt<B, T1, T2>(x1: T1, x2: T2) -> Tensor<B>
    where B: Backend, T1: IntoTensor<B>, T2: IntoTensor<B>
{
    map2(BinaryOperation::Gt, x1.into_tensor(), x2.into_tensor())
}


pub fn ge<B, T1, T2>(x1: T1, x2: T2) -> Tensor<B>
    where B: Backend, T1: IntoTensor<B>, T2: IntoTensor<B>
{
    map2(BinaryOperation::Ge, x1.into_tensor(), x2.into_tensor())
}


pub fn lt<B, T1, T2>(x1: T1, x2: T2) -> Tensor<B>
    where B: Backend, T1: IntoTensor<B>, T2: IntoTensor<B>
{
    map2(BinaryOperation::Lt, x1.into_tensor(), x2.into_tensor())
}


pub fn le<B, T1, T2>(x1: T1, x2: T2) -> Tensor<B>
    where B: Backend, T1: IntoTensor<B>, T2: IntoTensor<B>
{
    map2(BinaryOperation::Le, x1.into_tensor(), x2.into_tensor())
}


pub fn cond<B, T1, T2, T3>(x1: T1, x2: T2, x3: T3) -> Tensor<B>
    where B: Backend, T1: IntoTensor<B>, T2: IntoTensor<B>, T3: IntoTensor<B>
{
    map3(TernaryOperation::Cond, x1.into_tensor(), x2.into_tensor(), x3.into_tensor())
}


impl<B: Backend, T: IntoTensor<B>> ops::Add<T> for Tensor<B>
{
    type Output = Tensor<B>;
    fn add(self, x: T) -> Self::Output {
        add(self, x)
    }
}

impl<B: Backend, T: IntoTensor<B>> ops::Add<T> for &Tensor<B>
{
    type Output = Tensor<B>;
    fn add(self, x: T) -> Self::Output {
        add(self, x)
    }
}


impl<B: Backend, T: IntoTensor<B>> ops::Sub<T> for Tensor<B>
{
    type Output = Tensor<B>;
    fn sub(self, x: T) -> Self::Output {
        sub(self, x)
    }
}

impl<B: Backend, T: IntoTensor<B>> ops::Sub<T> for &Tensor<B>
{
    type Output = Tensor<B>;
    fn sub(self, x: T) -> Self::Output {
        sub(self, x)
    }
}


impl<B: Backend, T: IntoTensor<B>> ops::Div<T> for Tensor<B>
{
    type Output = Tensor<B>;
    fn div(self, x: T) -> Self::Output {
        div(self, x)
    }
}

impl<B: Backend, T: IntoTensor<B>> ops::Div<T> for &Tensor<B>
{
    type Output = Tensor<B>;
    fn div(self, x: T) -> Self::Output {
        div(self, x)
    }
}

impl<B: Backend, T: IntoTensor<B>> ops::Mul<T> for Tensor<B>
{
    type Output = Tensor<B>;
    fn mul(self, x: T) -> Self::Output {
        mul(self, x)
    }
}

impl<B: Backend, T: IntoTensor<B>> ops::Mul<T> for &Tensor<B>
{
    type Output = Tensor<B>;
    fn mul(self, x: T) -> Self::Output {
        mul(self, x)
    }
}

impl<B: Backend> ops::Neg for Tensor<B>
{
    type Output = Tensor<B>;
    fn neg(self) -> Self::Output {
        neg(self)
    }
}

impl<B: Backend> ops::Neg for &Tensor<B>
{
    type Output = Tensor<B>;
    fn neg(self) -> Self::Output {
        neg(self)
    }
}

