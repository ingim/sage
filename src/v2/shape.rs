use itertools::Itertools;
use smallvec::SmallVec;
use std::convert::TryInto;
use thiserror::Error;

pub type Array = SmallVec<[usize; 5]>;

pub fn display_comma(arr: &[usize]) -> String {
    arr.iter().map(|s| s.to_string()).join(", ")
}

#[derive(Error, Debug, Eq, PartialEq)]
pub enum ShapeError {
    #[error("size mismatch! expected {} but got {}.", .0, .1)]
    SizeMismatch(usize, usize),

    #[error("cannot infer the size")]
    InvalidInference,

    #[error("invalid shape extent {}, size should be larger than 0 or set to -1 for inference", .0)]
    InvalidExtent(isize),

    #[error("index out of range, expected index in range of {}..{}, but {} is given.", .low, .high, .index)]
    OutOfBounds {
        index: isize,
        low: isize,
        high: isize,
    },

    #[error("invalid index bound")]
    InvalidBound,

    #[error("invalid broadcast")]
    InvalidBroadcast,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    pub extents: Array,
    pub strides: Array,
    pub offset: usize,
}

impl Shape {
    pub fn default_strides(extents: &[usize]) -> Array {
        let size = extents.iter().product();
        extents
            .iter()
            .scan(size, |size, extent| {
                *size /= extent;
                Some(*size)
            })
            .collect()
    }

    pub fn new<E>(extents: E) -> Shape
    where
        E: Extent,
    {
        let extents = extents.to_arr(0).unwrap();
        let strides = Self::default_strides(&extents);
        Shape {
            extents,
            strides,
            offset: 0,
        }
    }

    pub fn num_axes(&self) -> usize {
        self.extents.len()
    }

    pub fn extents(&self) -> &[usize] {
        &self.extents
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn size(&self) -> usize {
        self.extents.iter().product()
    }

    pub fn translate_default(&self, index: usize) -> usize {
        let mut out_index = self.offset;
        let mut p = self.size();
        let mut rem = index;
        for i in 0..self.extents.len() {
            p /= self.extents[i];
            let c = rem / p;
            rem -= c * p;
            out_index += c * self.strides[i];
        }
        out_index
    }

    pub fn translate(&self, index: usize, reference: &Shape) -> usize {
        let mut out_index = self.offset;
        let mut rem = index - reference.offset;
        for i in 0..self.extents.len() {
            let c = rem / reference.strides[i];
            rem -= c * reference.strides[i];
            out_index += c * self.strides[i];
        }
        out_index
    }

    pub fn extract(&self, axes: &[usize], nested: bool) -> (Shape, Shape) {
        let not_axes = (0..self.extents.len()).filter(|a| !axes.contains(a));

        let inner = Shape {
            extents: axes.iter().map(|&a| self.extents[a]).collect(),
            strides: axes.iter().map(|&a| self.strides[a]).collect(),
            offset: if nested { 0 } else { self.offset },
        };

        let outer = Shape {
            extents: not_axes.clone().map(|a| self.extents[a]).collect(),
            strides: not_axes.map(|a| self.strides[a]).collect(),
            offset: self.offset,
        };
        // println!("orig: {:?}", &self);
        // println!("inner: {:?}", &inner);
        // println!("outer {:?}", &outer);

        (inner, outer)
    }

    pub fn has_default_strides(&self) -> bool {
        self.extents
            .iter()
            .scan(self.size(), |size, extent| {
                *size /= extent;
                Some(*size)
            })
            .zip(self.strides.iter())
            .all(|(s1, s2)| s1 == *s2)
    }

    // Whether the indices described by this memory layout is contiguous
    // Returns true if there are no indexing (or slicing) operations.
    // This method asks: Did user do any indexing (slicing) operations?
    pub fn is_contiguous(&self) -> bool {
        // max index == (unbroadcasted) theoretical max index
        let max_index = self
            .extents
            .iter()
            .zip(self.strides.iter())
            .map(|(extent, stride)| (extent - 1) * stride)
            .sum::<usize>();

        let t_max_index = self
            .extents
            .iter()
            .zip(self.strides.iter())
            .filter(|(_, &stride)| stride > 0)
            .map(|(&extent, _)| extent)
            .product::<usize>()
            - 1;

        max_index == t_max_index
    }

    // one-to-one correspondence
    // This method asks: Did user do any broadcasting operations?
    pub fn is_bijective(&self) -> bool {
        self.strides.iter().all(|&stride| stride > 0)
    }

    // This method asks: did the user used any reshape operations?
    pub fn is_ordered(&self) -> bool {
        self.strides
            .iter()
            .filter(|&stride| *stride > 0)
            .is_sorted_by(|&a, &b| Some(b.cmp(a)))
    }

    pub fn remove(&self, axis: usize) -> Shape {
        let mut shape = self.clone();
        shape.extents.remove(axis);
        shape.strides.remove(axis);
        shape
    }

    pub fn insert(&self, axis: usize) -> Shape {
        let mut shape = self.clone();
        shape.extents.insert(axis, 1);
        shape.strides.insert(axis, 0);
        shape
    }

    pub fn swap(&self, axis1: usize, axis2: usize) -> Shape {
        let mut shape = self.clone();
        shape.extents.swap(axis1, axis2);
        shape.strides.swap(axis1, axis2);
        shape
    }

    pub fn permute(&self, axes: &[usize]) -> Shape {
        let (new_extents, new_strides) = axes
            .iter()
            .map(|axis| (self.extents[*axis], self.strides[*axis]))
            .unzip();
        Shape {
            extents: new_extents,
            strides: new_strides,
            offset: self.offset,
        }
    }

    pub fn select(&self, index: usize, axis: usize) -> Shape {
        if self.num_axes() <= axis {
            panic!("axis out of bounds");
        }

        if self.extents[axis] <= index {
            panic!("index out of bounds");
        }
        let mut shape = self.clone();
        shape.offset += shape.strides[axis] * index;
        shape.remove(axis)
    }

    pub fn select_range(&self, index_start: usize, index_end: usize, axis: usize) -> Shape {
        if self.extents[axis] <= index_end {
            panic!("index out of bounds");
        }
        let mut shape = self.clone();
        shape.offset += self.strides[axis] * index_start;
        shape.extents[axis] = index_end - index_start + 1;
        shape
    }

    pub fn expand(&self, extents: &[usize]) -> Shape {
        if extents.len() < self.extents.len() {
            panic!("target shape must be larger than the broadcasted shape");
        }

        let mut new_extents = self.extents.clone();
        let mut new_strides = self.strides.clone();

        // Let's say that we are broadcasting
        // (3, 1, 5) to (2, 1, 3, 9, 5)

        // First, we add padding so that
        // (3, 1, 5) -> (1, 1, 3, 1, 5)
        for _ in 0..(extents.len() - self.extents.len()) {
            new_extents.insert(0, 1);
            new_strides.insert(0, 0);
        }

        // Next, we update extents while checking its validity
        for ((new_extent, extent), new_stride) in new_extents
            .iter_mut()
            .zip(extents.iter())
            .zip(new_strides.iter_mut())
        {
            if *new_extent != *extent {
                // for broadcasted axes, 'mute' them by set its stride to zero
                if *new_extent == 1 {
                    *new_extent = *extent;
                    *new_stride = 0;
                } else {
                    panic!("invalid broadcast... target shape should be larger.");
                }
            }
        }

        Shape {
            extents: new_extents,
            strides: new_strides,
            offset: self.offset,
        }
    }

    pub fn iter(&self) -> IndexIter {
        IndexIter::new(self)
    }
}

pub struct IndexIter {
    shape: Shape,
    index: usize,
    len: usize,
}

impl IndexIter {
    pub fn new(layout: &Shape) -> Self {
        IndexIter {
            shape: layout.clone(),
            index: 0,
            len: layout.size(),
        }
    }
}

impl Iterator for IndexIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.len {
            let t_index = self.shape.translate_default(self.index);
            self.index += 1;
            Some(t_index)
        } else {
            None
        }
    }
}

impl DoubleEndedIterator for IndexIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index < self.len {
            let t_index = self.shape.translate_default(self.len - 1);
            self.len -= 1;
            Some(t_index)
        } else {
            None
        }
    }
}

impl ExactSizeIterator for IndexIter {
    fn len(&self) -> usize {
        self.len
    }
}

pub trait Size: Extent {
    fn needs_infer(&self) -> bool;
    fn to_usize(&self) -> Result<usize, ShapeError>;
}

pub trait Axis: Axes {
    fn to_usize(&self, bound: usize) -> Result<usize, ShapeError>;
}

pub trait Extent {
    fn to_arr(&self, size: usize) -> Result<Array, ShapeError>;
}

pub trait Axes {
    fn to_arr(&self, bound: usize) -> Result<Array, ShapeError>;
}

pub trait SizedExtent<const N: usize> {
    fn to_arr(&self) -> [usize; N];
    fn at(&self, idx: usize) -> usize;
}

pub fn union<S1, S2>(shape1: S1, shape2: S2) -> Result<Array, ShapeError>
where
    S1: Extent,
    S2: Extent,
{
    let shape1 = shape1.to_arr(0).unwrap();
    let shape2 = shape2.to_arr(0).unwrap();

    if shape1 == shape2 {
        Ok(shape1)
    }
    // Do union
    else {
        let (longer, shorter) = if shape1.len() > shape2.len() {
            (shape1, shape2)
        } else {
            (shape2, shape1)
        };

        let len = longer.len() - shorter.len();
        let mut u = shorter;

        for i in 0..len {
            u.insert(i, longer[i]);
        }

        for (a, b) in u.iter_mut().zip(longer.iter()) {
            if *a != *b {
                if *a == 1 {
                    *a = *b;
                } else if *b != 1 {
                    return Err(ShapeError::InvalidBroadcast);
                }
            }
        }
        Ok(u)
    }
}

fn axes_to_arr<A>(axes: &[A], bound: usize) -> Result<Array, ShapeError>
where
    A: Axis,
{
    axes.iter().map(|i| i.to_usize(bound)).try_collect()
}

fn extents_to_arr<E>(extents: &[E], size: usize) -> Result<Array, ShapeError>
where
    E: Size,
{
    let mut use_infer = false;
    let mut infer_idx = 0;

    let mut expected_size = 1;
    let mut vec = Array::with_capacity(extents.len());

    for (i, extent) in extents.iter().enumerate() {
        if extent.needs_infer() {
            if !use_infer {
                use_infer = true;
                infer_idx = i;
            } else {
                return Err(ShapeError::InvalidInference);
            }
        } else {
            let e = extent.to_usize()?;
            vec.push(e);
            expected_size *= e;
        }
    }

    if !use_infer && expected_size != size && size > 0 {
        return Err(ShapeError::SizeMismatch(size, expected_size));
    }

    if use_infer && size == 0 {
        return Err(ShapeError::InvalidInference);
    }

    if use_infer && size % expected_size != 0 {
        return Err(ShapeError::InvalidInference);
    }
    if use_infer {
        vec.insert(infer_idx, size / expected_size)
    }
    Ok(vec)
}

macro_rules! impl_extent_unsigned {
    ($ty:ty) => {
        impl Size for $ty {
            fn needs_infer(&self) -> bool {
                false
            }

            fn to_usize(&self) -> Result<usize, ShapeError> {
                if *self > 0 {
                    Ok(*self as usize)
                } else {
                    Err(ShapeError::InvalidExtent(*self as isize))
                }
            }
        }

        impl Extent for $ty {
            fn to_arr(&self, size: usize) -> Result<Array, ShapeError> {
                extents_to_arr(&[*self], size)
            }
        }
    };
}

macro_rules! impl_extent_signed {
    ($ty:ty) => {
        impl Size for $ty {
            fn needs_infer(&self) -> bool {
                *self == -1
            }

            fn to_usize(&self) -> Result<usize, ShapeError> {
                if *self > 0 {
                    Ok(*self as usize)
                } else {
                    Err(ShapeError::InvalidExtent((*self).try_into().unwrap()))
                }
            }
        }

        impl Extent for $ty {
            fn to_arr(&self, size: usize) -> Result<Array, ShapeError> {
                extents_to_arr(&[*self], size)
            }
        }
    };
}

macro_rules! impl_axis_unsigned {
    ($ty:ty) => {
        impl Axis for $ty {
            fn to_usize(&self, bound: usize) -> Result<usize, ShapeError> {
                if bound < 1 {
                    return Err(ShapeError::InvalidBound);
                }
                let axis = *self as usize;
                if axis < bound {
                    Ok(axis)
                } else {
                    Err(ShapeError::OutOfBounds {
                        index: axis as isize,
                        low: -(bound as isize),
                        high: (bound - 1) as isize,
                    })
                }
            }
        }
        impl Axes for $ty {
            fn to_arr(&self, bound: usize) -> Result<Array, ShapeError> {
                axes_to_arr(&[*self], bound)
            }
        }
    };
}

macro_rules! impl_axis_signed {
    ($ty:ty) => {
        impl Axis for $ty {
            fn to_usize(&self, bound: usize) -> Result<usize, ShapeError> {
                if bound < 1 {
                    return Err(ShapeError::InvalidBound);
                }
                let axis = *self as isize;
                let axis = if axis >= 0 {
                    axis
                } else {
                    axis + bound as isize
                } as usize;

                if axis < bound {
                    Ok(axis)
                } else {
                    Err(ShapeError::OutOfBounds {
                        index: *self as isize,
                        low: -(bound as isize),
                        high: (bound - 1) as isize,
                    })
                }
            }
        }
        impl Axes for $ty {
            fn to_arr(&self, bound: usize) -> Result<Array, ShapeError> {
                axes_to_arr(&[*self], bound)
            }
        }
    };
}

impl_extent_unsigned!(u32);
impl_extent_unsigned!(usize);

impl_extent_signed!(i32);
impl_extent_signed!(isize);

impl_axis_unsigned!(u32);
impl_axis_unsigned!(usize);

impl_axis_signed!(i32);
impl_axis_signed!(isize);

impl<T, const C: usize> Extent for [T; C]
where
    T: Size,
{
    fn to_arr(&self, size: usize) -> Result<Array, ShapeError> {
        extents_to_arr(self, size)
    }
}

impl<'a, T> Extent for &'a [T]
where
    T: Size,
{
    fn to_arr(&self, size: usize) -> Result<Array, ShapeError> {
        extents_to_arr(self, size)
    }
}

impl<T> Extent for Vec<T>
where
    T: Size,
{
    fn to_arr(&self, size: usize) -> Result<Array, ShapeError> {
        extents_to_arr(self, size)
    }
}

impl<T> Extent for &Vec<T>
where
    T: Size,
{
    fn to_arr(&self, size: usize) -> Result<Array, ShapeError> {
        extents_to_arr(self, size)
    }
}

impl Extent for Array {
    fn to_arr(&self, size: usize) -> Result<Array, ShapeError> {
        extents_to_arr(self.as_slice(), size)
    }
}

impl Extent for &Array {
    fn to_arr(&self, size: usize) -> Result<Array, ShapeError> {
        extents_to_arr(self.as_slice(), size)
    }
}

impl<T, const C: usize> Axes for [T; C]
where
    T: Axis,
{
    fn to_arr(&self, bound: usize) -> Result<Array, ShapeError> {
        axes_to_arr(self, bound)
    }
}

impl<'a, T> Axes for &'a [T]
where
    T: Axis,
{
    fn to_arr(&self, bound: usize) -> Result<Array, ShapeError> {
        axes_to_arr(self, bound)
    }
}

impl<T> Axes for Vec<T>
where
    T: Axis,
{
    fn to_arr(&self, bound: usize) -> Result<Array, ShapeError> {
        axes_to_arr(self, bound)
    }
}

impl<T> Axes for &Vec<T>
where
    T: Axis,
{
    fn to_arr(&self, bound: usize) -> Result<Array, ShapeError> {
        axes_to_arr(self, bound)
    }
}

impl Axes for Array {
    fn to_arr(&self, bound: usize) -> Result<Array, ShapeError> {
        axes_to_arr(self.as_slice(), bound)
    }
}

impl Axes for &Array {
    fn to_arr(&self, bound: usize) -> Result<Array, ShapeError> {
        axes_to_arr(self.as_slice(), bound)
    }
}

pub static all_axes: AllAxes = AllAxes {};

#[derive(Copy, Clone)]
pub struct AllAxes {}

impl Axes for AllAxes {
    fn to_arr(&self, bound: usize) -> Result<Array, ShapeError> {
        Ok((0..bound).collect::<Array>())
    }
}

impl<const N: usize> SizedExtent<N> for usize {
    fn to_arr(&self) -> [usize; N] {
        [*self; N]
    }

    fn at(&self, _: usize) -> usize {
        *self
    }
}

impl<const N: usize> SizedExtent<N> for [usize; N] {
    fn to_arr(&self) -> [usize; N] {
        *self
    }

    fn at(&self, idx: usize) -> usize {
        self[idx % N]
    }
}

#[cfg(test)]
mod tests {
    use crate::v2::shape::{axes_to_arr, extents_to_arr, union, Axis, ShapeError, Size};

    fn axis_to_usize<A: Axis>(a: A, bound: usize) -> Result<usize, ShapeError> {
        a.to_usize(bound)
    }

    fn extent_to_usize<E: Size>(e: E) -> Result<usize, ShapeError> {
        e.to_usize()
    }

    fn extent_needs_infer<E: Size>(e: E) -> bool {
        e.needs_infer()
    }

    #[test]
    fn test_axis() {
        assert_eq!(axis_to_usize(-3_isize, 3).unwrap(), 0);
        assert_eq!(axis_to_usize(-2_i32, 3).unwrap(), 1);
        assert_eq!(axis_to_usize(0_usize, 3).unwrap(), 0);
        assert_eq!(axis_to_usize(1_u32, 3).unwrap(), 1);
    }

    #[test]
    fn test_axis_err_oob() {
        assert_eq!(axis_to_usize(0, 0).expect_err(""), ShapeError::InvalidBound);
        assert_eq!(
            axis_to_usize(-4, 3).expect_err(""),
            ShapeError::OutOfBounds {
                index: -4,
                low: -3,
                high: 2,
            }
        );
        assert_eq!(
            axis_to_usize(4, 4).expect_err(""),
            ShapeError::OutOfBounds {
                index: 4,
                low: -4,
                high: 3,
            }
        );
    }

    #[test]
    fn test_extent() {
        assert_eq!(extent_to_usize(1_usize).unwrap(), 1);
        assert_eq!(extent_to_usize(1_u32).unwrap(), 1);
        assert_eq!(extent_to_usize(1_isize).unwrap(), 1);
        assert_eq!(extent_to_usize(1_i32).unwrap(), 1);
    }

    #[test]
    fn test_extent_err_invalid() {
        assert_eq!(
            extent_to_usize(-1).expect_err(""),
            ShapeError::InvalidExtent(-1)
        );
        assert_eq!(
            extent_to_usize(0).expect_err(""),
            ShapeError::InvalidExtent(0)
        );
    }

    #[test]
    fn test_extent_needs_infer() {
        assert_eq!(extent_needs_infer(-2), false);
        assert_eq!(extent_needs_infer(-1), true);
        assert_eq!(extent_needs_infer(0), false);
        assert_eq!(extent_needs_infer(1), false);
    }

    #[test]
    fn test_axes_to_vec() {
        assert_eq!(
            axes_to_arr(&[-3, -2, -1, 0, 1, 2], 3).unwrap().to_vec(),
            vec![0, 1, 2, 0, 1, 2]
        );
    }

    #[test]
    fn test_axes_to_vec_err() {
        assert_eq!(
            axes_to_arr(&[-4, -2, -1, 0, 1, 2], 3).expect_err(""),
            ShapeError::OutOfBounds {
                index: -4,
                low: -3,
                high: 2,
            }
        );
    }

    #[test]
    fn test_shape_to_vec() {
        assert_eq!(
            extents_to_arr(&[2, 3, 4], 0).unwrap().to_vec(),
            vec![2, 3, 4]
        );
        assert_eq!(
            extents_to_arr(&[2, 3, 4], 24).unwrap().to_vec(),
            vec![2, 3, 4]
        );
        assert_eq!(
            extents_to_arr(&[-1, 3, 4], 24).unwrap().to_vec(),
            vec![2, 3, 4]
        );
        assert_eq!(
            extents_to_arr(&[2, -1, 4], 24).unwrap().to_vec(),
            vec![2, 3, 4]
        );
        assert_eq!(
            extents_to_arr(&[2, 3, -1], 24).unwrap().to_vec(),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn test_shape_to_vec_err() {
        // invalid extent size
        assert_eq!(
            extents_to_arr(&[1, 3, -4], 24).expect_err(""),
            ShapeError::InvalidExtent(-4)
        );
        assert_eq!(
            extents_to_arr(&[1, 0, 4], 24).expect_err(""),
            ShapeError::InvalidExtent(0)
        );

        // size mismatch
        assert_eq!(
            extents_to_arr(&[1, 3, 4], 24).expect_err(""),
            ShapeError::SizeMismatch(24, 12)
        );
        assert_eq!(
            extents_to_arr(&[1, 3, 4], 23).expect_err(""),
            ShapeError::SizeMismatch(23, 12)
        );
        assert_eq!(
            extents_to_arr(&[1, 3, 4], 25).expect_err(""),
            ShapeError::SizeMismatch(25, 12)
        );

        // infer two times
        assert_eq!(
            extents_to_arr(&[-1, -1, 4], 24).expect_err(""),
            ShapeError::InvalidInference
        );

        // use infer but no specified size
        assert_eq!(
            extents_to_arr(&[-1, 3, 4], 0).expect_err(""),
            ShapeError::InvalidInference
        );

        // not divisible infer size
        assert_eq!(
            extents_to_arr(&[-1, 3, 4], 25).expect_err(""),
            ShapeError::InvalidInference
        );
    }

    #[test]
    fn test_union() {
        assert_eq!(union([1, 3, 4], [1, 3, 4]).unwrap().to_vec(), vec![1, 3, 4]);
        assert_eq!(
            union([5, 6, 1, 3, 4], [1, 3, 4]).unwrap().to_vec(),
            vec![5, 6, 1, 3, 4]
        );
        assert_eq!(
            union([5, 6, 1, 3, 4], [7, 3, 4]).unwrap().to_vec(),
            vec![5, 6, 7, 3, 4]
        );
        assert_eq!(
            union([1, 2, 1, 2, 1, 2], [2, 1, 2, 1, 2, 1])
                .unwrap()
                .to_vec(),
            vec![2, 2, 2, 2, 2, 2]
        );
    }
}
