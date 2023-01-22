use crate::v2::backend::Backend;
use crate::v2::ir::{Graph, NodeId};
use crate::v2::shape::{Array, Extent, Shape};
use crate::v2::tensor::{IntoTensor, Operator, Tensor};

pub mod map;


#[derive(Clone, Debug)]
pub enum LayoutOperator {
    Expand(Expand),
}

// copy
#[derive(Clone, Debug)]
struct Reshape {
    shape: Shape,
}

// no-copy
#[derive(Clone, Debug)]
pub struct Expand {
    shape: Shape,
}

#[derive(Clone, Debug)]
struct Permute {
    perm: Array,
}

#[derive(Clone, Debug)]
struct Squeeze {
    axis: usize,
}

#[derive(Clone, Debug)]
struct Unsqueeze {
    axis: usize,
}

#[derive(Clone, Debug)]
struct Slice {
    axis: usize,
    start: usize,
    end: usize,
}

#[derive(Clone, Debug)]
struct Unslice {
    axis: usize,
    start: usize,
    end: usize,
}

#[derive(Clone, Debug)]
struct Gather {
    axis: usize,
}

#[derive(Clone, Debug)]
struct Scatter {
    axis: usize,
}

pub fn ceil_div(a: usize, b: usize) -> usize {
    1 + ((a - 1) / b)
}

pub fn ceil(a: usize, b: usize) -> usize {
    ceil_div(a, b) * b
}

pub fn div_up(a: usize, b: usize) -> usize {
    (a + (b - 1)) / b
}

pub fn inc_arr<const N: usize>(start: usize) -> [usize; N] {
    let mut x = [0; N];
    for i in 0..N {
        x[i] = i + start;
    }
    x
}

pub fn view<B, T, E>(x: T, extents: E) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>, E: Extent
{
    let x = x.into_tensor();
    let extents = extents.to_arr(x.size()).unwrap();
    reshape(x, Shape::new(&extents))
}

pub fn reshape<B, T>(x: T, shape: Shape) -> Tensor<B>
    where B: Backend, T: IntoTensor<B>
{
    let x = x.into_tensor().contiguous();

    if x.size() != shape.size() {
        panic!("incompatible size");
    }

    Tensor::from_op(
        Reshape {
            shape: shape.clone()
        },
        [x],
        shape,
    )
}

// always creates new tensor
impl<B: Backend> Operator<1, B> for Reshape {
    fn grad(&self, x: &[Tensor<B>; 1], _: &Tensor<B>, gy: &Tensor<B>) -> [Option<Tensor<B>>; 1] {
        [Some(reshape(gy, x[0].shape().clone()))]
    }

    fn build(&self, x: [NodeId; 1], g: &mut Graph) -> NodeId {
        // g.reshape(x[0], self.shape.clone())
        todo!()
    }
}

pub fn expand<V, E>(x: V, extents: E) -> Fun
    where
        V: ToFun,
        E: Extent,
{
    let x = x.to_fun();
    let extents = extents.to_arr(0).unwrap();

    Fun::from_unary_op(
        Expand {
            input: [x.desc().clone()],
            output: TensorDesc {
                shape: x.shape().expand(&extents),
                data_type: x.data_type(),
            },
        },
        x,
    )
}

impl Compose<1> for Expand {
    fn input(&self) -> &[TensorDesc; 1] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, x: [&Fun; 1], _: &Fun, gy: &Fun) -> [Option<Fun>; 1] {
        // match dims from the backwards

        let surplus = gy.rank() - x[0].rank();
        let mut axes = Array::new();

        for i in 0..surplus {
            axes.push(i);
        }

        for i in 0..x[0].rank() {
            if gy.extent(i + surplus) != x[0].extent(i) {
                axes.push(i + surplus);
            }
        }

        // println!("gy {:?}", gy.extents());
        // println!("expected gx {:?}", x.extents());
        // println!("actual gx {:?}", gy.sum(&axes, false).extents());

        [Some(gy.sum(axes, false).view(self.input[0].extents()))]
    }

    fn compute(&self, x: [&Tensor; 1], _: &mut Context) -> Result<Tensor, Error> {
        Ok(x[0].expand(self.output().shape.extents()))
    }

    fn cat(&self) -> Category {
        Category::Layout(LayoutOperator::Expand(self.clone()))
    }
}

pub fn tr<V>(x: V) -> Fun
    where
        V: ToFun,
{
    transpose(x, 0, 1)
}

pub fn transpose<V, A1, A2>(x: V, axis1: A1, axis2: A2) -> Fun
    where
        V: ToFun,
        A1: Axis,
        A2: Axis,
{
    let x = x.to_fun();
    let rank = x.rank();

    let axis1 = axis1.to_usize(rank).unwrap();
    let axis2 = axis2.to_usize(rank).unwrap();

    let axes = (0..rank)
        .map(|i| {
            if i == axis1 {
                axis2
            } else if i == axis2 {
                axis1
            } else {
                i
            }
        })
        .collect::<Array>();

    permute(x, axes)
}

pub fn permute<V, A>(x: V, axes: A) -> Fun
    where
        V: ToFun,
        A: Axes,
{
    let x = x.to_fun();
    let axes = axes.to_arr(x.rank()).unwrap();

    Fun::from_unary_op(
        Permute {
            input: [x.desc().clone()],
            output: TensorDesc {
                shape: x.shape().permute(axes.as_slice()),
                data_type: x.data_type(),
            },
            perm: axes,
        },
        x,
    )
}

impl Compose<1> for Permute {
    fn input(&self) -> &[TensorDesc; 1] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, _: [&Fun; 1], _: &Fun, gy: &Fun) -> [Option<Fun>; 1] {
        // [A, B, C, D] -> permute(2, 1, 4, 3) -> [B, A, D, C]
        // [B, A, D, C] -> permute(2, 1, 4, 3) -> [A, B, C, D]

        [Some(permute(gy, self.perm.as_slice()))]
    }

    fn compute(&self, x: [&Tensor; 1], _: &mut Context) -> Result<Tensor, Error> {
        Ok(x[0].permute(self.perm.as_slice()))
    }
}

pub fn squeeze<V, A>(x: V, axis: A) -> Fun
    where
        V: ToFun,
        A: Axis,
{
    let mut x = x.to_fun();
    let axis = axis.to_usize(x.rank()).unwrap();

    Fun::from_unary_op(
        Squeeze {
            input: [x.desc().clone()],
            output: TensorDesc {
                shape: x.shape().remove(axis),
                data_type: x.data_type(),
            },
            axis,
        },
        x,
    )
}

impl Compose<1> for Squeeze {
    fn input(&self) -> &[TensorDesc; 1] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, _: [&Fun; 1], _: &Fun, gy: &Fun) -> [Option<Fun>; 1] {
        [Some(unsqueeze(gy, self.axis))]
    }

    fn compute(&self, x: [&Tensor; 1], _: &mut Context) -> Result<Tensor, Error> {
        Ok(x[0].squeeze_axis(self.axis))
    }
}

pub fn unsqueeze<V, A>(x: V, axis: A) -> Fun
    where
        V: ToFun,
        A: Axis,
{
    let x = x.to_fun();
    let axis = axis.to_usize(x.rank() + 1).unwrap();

    Fun::from_unary_op(
        Unsqueeze {
            input: [x.desc().clone()],
            output: TensorDesc {
                shape: x.shape().insert(axis),
                data_type: x.data_type(),
            },
            axis,
        },
        x,
    )
}

impl Compose<1> for Unsqueeze {
    fn input(&self) -> &[TensorDesc; 1] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, _: [&Fun; 1], _: &Fun, gy: &Fun) -> [Option<Fun>; 1] {
        [Some(squeeze(gy, self.axis))]
    }

    fn compute(&self, x: [&Tensor; 1], _: &mut Context) -> Result<Tensor, Error> {
        Ok(x[0].expand_axis(self.axis))
    }
}

pub fn slice<V, A>(x: V, axis: A, start: usize, end: usize) -> Fun
    where
        V: ToFun,
        A: Axis,
{
    let x = x.to_fun();
    let axis = axis.to_usize(x.rank()).unwrap();

    Fun::from_unary_op(
        Slice {
            input: [x.desc().clone()],
            output: TensorDesc {
                shape: x.shape().select_range(start, end - 1, axis),
                data_type: x.data_type(),
            },
            axis,
            start,
            end,
        },
        x,
    )
}

impl Compose<1> for Slice {
    fn input(&self) -> &[TensorDesc; 1] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, x: [&Fun; 1], _: &Fun, gy: &Fun) -> [Option<Fun>; 1] {
        // scatter add
        [Some(unslice(
            gy,
            self.axis,
            self.start,
            x[0].extent(self.axis),
        ))]
    }

    fn compute(&self, x: [&Tensor; 1], _: &mut Context) -> Result<Tensor, Error> {
        Ok(x[0].slice(self.start, self.end, self.axis))
    }
}

pub fn unslice<V, A>(x: V, axis: A, start: usize, end: usize) -> Fun
    where
        V: ToFun,
        A: Axis,
{
    let x = x.to_fun();
    let axis = axis.to_usize(x.rank()).unwrap();

    assert!(end >= start + x.extent(axis));

    let mut extents = x.extents().to_vec();
    extents[axis] = end;

    Fun::from_unary_op(
        Unslice {
            input: [x.desc().clone()],
            output: TensorDesc {
                shape: Shape::new(extents),
                data_type: x.data_type(),
            },
            axis,
            start,
            end,
        },
        x,
    )
}

impl Compose<1> for Unslice {
    fn input(&self) -> &[TensorDesc; 1] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, x: [&Fun; 1], _: &Fun, gy: &Fun) -> [Option<Fun>; 1] {
        [Some(slice(
            gy,
            self.axis,
            self.start,
            self.start + x[0].extent(self.axis),
        ))]
    }

    fn compute(&self, x: [&Tensor; 1], ctx: &mut Context) -> Result<Tensor, Error> {
        let x = x[0];
        let mut extents = x.extents().to_vec();
        extents[self.axis] = self.end;

        let y = Tensor::uninit2(self.output(), ctx)?; // all defaults to gpu tensors
        y.fill_zero();

        let stride_ref = Shape::default_strides(x.extents());

        let data_type = x.data_type().opencl();
        // computer.submit(ctx.backend.unslice())
        let (idx_c, is_direct) = translate_id(
            "gid",
            &stride_ref,
            &["idx_x", "idx_y"],
            &[x.strides(), y.strides()],
            &[x.offset(), y.offset() + self.start * y.strides()[self.axis]],
        );

        let p = ctx.get_program(format!(
            r#"
        __kernel void unslice(
            __global const {data_type} *x,
            __global {data_type} *y) {{
            uint gid = get_global_id(0);
            {idx_c}
            y[idx_y] = x[idx_x];
        }} "#
        ));

        p.kernel("unslice")
            .arg_tensor(x)
            .arg_tensor(&y)
            .global_work_size(x.size())
            .launch()
            .map_err(Error::Device)?;

        Ok(y)
    }
}

pub fn gather<V1, V2, A>(x: V1, ind: V2, axis: A) -> Fun
    where
        V1: ToFun,
        V2: ToFun,
        A: Axis,
{
    let x = x.to_fun();
    let ind = ind.to_fun();
    let axis = axis.to_usize(x.rank()).unwrap();

    assert_eq!(x.rank(), ind.rank());

    // (164, 100) g0 (64, 100) -> (64, 1)
    assert!((0..x.rank()).any(|i| i == axis || x.extent(i) >= ind.extent(i)));

    Fun::from_binary_op(
        Gather {
            input: [x.desc().clone(), ind.desc().clone()],
            output: TensorDesc {
                shape: Shape::new(ind.extents()),
                data_type: x.data_type(),
            },
            axis,
        },
        x,
        ind,
    )
}

impl Compose<2> for Gather {
    fn input(&self) -> &[TensorDesc; 2] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, x: [&Fun; 2], _: &Fun, gy: &Fun) -> [Option<Fun>; 2] {
        [Some(scatter(gy, x[1], self.axis, x[0].extents())), None]
    }

    fn compute(&self, x: [&Tensor; 2], ctx: &mut Context) -> Result<Tensor, Error> {
        let strides = display_comma(x[0].strides());
        let strides_idx = display_comma(x[1].strides());

        let strides_d = display_comma(Shape::default_strides(x[1].extents()).as_slice());
        let ndim = x[0].rank(); // == ind.order();
        let axis = self.axis;
        let data_type = x[0].data_type().opencl();

        let y = Tensor::uninit2(self.output(), ctx)?; // all defaults to gpu tensors

        let (gather_id_c, _) = translate_id(
            "gid",
            &Shape::default_strides(x[1].extents()),
            &["gather_idx"],
            &[x[1].strides()],
            &[0],
        );

        let p = ctx.get_program(format!(
            r#"
            __kernel void gather(
                __global const {data_type} *x,
                __private const int x_offset,
                __global const uint *ind,
                __private const int ind_offset,
                __global {data_type} *y,
                __private const int y_offset) {{

                uint gid = get_global_id(0);

                uint idx = x_offset;

                {gather_id_c}
                uint target_idx = ind[gather_idx + ind_offset];

                uint strides[{ndim}] = {{{strides}}};
                uint strides_d[{ndim}] = {{{strides_d}}};
                uint rem2 = gid;
                for (uint i = 0; i < {ndim}; ++i) {{
                    uint c = rem2 / strides_d[i];
                    rem2 -= c * strides_d[i];
                    if (i == {axis} ) {{
                        idx += target_idx * strides[i];
                    }} else {{
                        idx += c * strides[i];
                    }}
                }}
                y[gid + y_offset] = x[idx];
             }} "#
        ));

        p.kernel("gather")
            .arg_tensor(x[0])
            .arg(x[0].offset() as i32)
            .arg_tensor(x[1])
            .arg(x[1].offset() as i32)
            .arg_tensor(&y)
            .arg(y.offset() as i32)
            .global_work_size(y.size())
            .launch()
            .map_err(Error::Device)?;
        Ok(y)
    }
}

pub fn scatter<V1, V2, A, E>(x: V1, ind: V2, axis: A, extents: E) -> Fun
    where
        V1: ToFun,
        V2: ToFun,
        A: Axis,
        E: Extent,
{
    let x = x.to_fun();
    let ind = ind.to_fun();
    let axis = axis.to_usize(x.rank()).unwrap();
    let extents = extents.to_arr(0).unwrap();

    assert_eq!(extents.len(), x.rank());
    assert_eq!(x.rank(), ind.rank());

    // (64, 1) -> (64, 1) with (64, 10)
    assert!((0..x.rank())
        .any(|i| { i == axis || (x.extent(i) >= ind.extent(i) && extents[i] >= x.extent(i)) }));

    Fun::from_binary_op(
        Scatter {
            input: [x.desc().clone(), ind.desc().clone()],
            output: TensorDesc::new(extents, x.data_type()),
            axis,
        },
        x,
        ind,
    )
}

impl Compose<2> for Scatter {
    fn input(&self) -> &[TensorDesc; 2] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, x: [&Fun; 2], _: &Fun, gy: &Fun) -> [Option<Fun>; 2] {
        [Some(gather(gy, x[1], self.axis)), None]
    }

    fn compute(&self, x: [&Tensor; 2], ctx: &mut Context) -> Result<Tensor, Error> {
        let data_type = x[0].data_type();
        let strides_src = display_comma(x[0].strides());
        let strides_dst = display_comma(self.output.shape().strides());
        let strides_d = display_comma(x[1].strides());

        let ndim = self.output.shape().num_axes();
        let axis = self.axis;

        let (scatter_id_c, _) = translate_id(
            "gid",
            &Shape::default_strides(x[1].extents()),
            &["scatter_idx"],
            &[x[1].strides()],
            &[0],
        );

        let y = Tensor::uninit2(self.output(), ctx)?; // all defaults to gpu tensors
        y.fill_zero();

        let p = ctx.get_program(format!(
            r#"
        __kernel void scatter(__global const {data_type} *x,
            __private const int x_offset,
            __global const uint *ind,
            __private const int ind_offset,
            __global {data_type} *y,
            __private const int y_offset) {{

            uint gid = get_global_id(0);
            uint idx_in = x_offset;
            uint idx_out = y_offset;

            uint strides_dst[{ndim}] = {{{strides_dst}}};
            uint strides_src[{ndim}] = {{{strides_src}}};
            uint strides_d[{ndim}] = {{{strides_d}}};

            {scatter_id_c}

            uint target_idx = ind[scatter_idx + ind_offset];

            uint rem2 = gid;
            for (uint i = 0; i < {ndim}; ++i) {{
                uint c = rem2 / strides_d[i];
                rem2 -= c * strides_d[i];
                idx_in += c * strides_src[i];
                if (i == {axis}) {{
                    idx_out += target_idx * strides_dst[i];
                }} else {{
                    idx_out += c * strides_dst[i];
                }}
            }}
            y[idx_out + y_offset] += x[idx_in];
        }}
        "#
        ));

        p.kernel("scatter")
            .arg_tensor(x[0])
            .arg(x[0].offset() as i32)
            .arg_tensor(x[1])
            .arg(x[1].offset() as i32)
            .arg_tensor(&y)
            .arg(y.offset() as i32)
            .global_work_size(x[1].size())
            .launch()
            .map_err(Error::Device)?;

        Ok(y)
    }
}

pub fn concat<V, I, A>(x: I, axis: A) -> Fun
    where
        V: ToFun,
        I: IntoIterator<Item=V>,
        A: Axis,
{
    let x: Vec<Fun> = x.into_iter().map(|v| v.to_fun()).collect_vec();
    let axis = axis.to_usize(x[0].rank()).unwrap();

    let mut concat_size = x[0].extent(axis);

    for v in x.iter().skip(1) {
        if v.rank() != x[0].rank()
            || v.extents()[..axis] != x[0].extents()[..axis]
            || v.extents()[(axis + 1)..] != x[0].extents()[(axis + 1)..]
        {
            panic!("not in the same shape");
        }
        concat_size += v.extent(axis);
    }

    let mut extents = x[0].extents().to_vec();
    extents[axis] = concat_size;

    Fun::from_variadic_op(
        Concat {
            input: x.iter().map(|v| v.desc()).cloned().collect_vec(),
            output: TensorDesc {
                shape: Shape::new(extents),
                data_type: x[0].data_type(),
            },
            axis,
        },
        x,
    )
}

impl VariadicCompose for Concat {
    fn input(&self) -> &[TensorDesc] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, x: &[Fun], _: &Fun, gy: &Fun) -> Vec<Option<Fun>> {
        let mut i = 0;
        let mut g = Vec::new();
        for e in x.iter().map(|v| v.extents()[self.axis]) {
            g.push(Some(slice(gy, self.axis, i, i + e)));
            i += e;
        }
        g
    }

    fn compute(&self, x: &[Tensor], ctx: &mut Context) -> Result<Tensor, Error> {
        let data_type = x[0].data_type();

        //let y = Tensor::zeros(self.output().shape().extents()).to_device(ctx); // all defaults to gpu tensors
        let y = Tensor::uninit2(self.output(), ctx)?;
        y.fill_zero();

        let mut i = 0;

        for v in x {
            let e = v.extent(self.axis);

            let y_sub_shape = y.shape().select_range(i, i + e - 1, self.axis);
            i += e;

            let stride_ref = Shape::default_strides(v.extents());

            let (idx_c, is_direct) = translate_id(
                "gid",
                &stride_ref,
                &["idx_in", "idx_out"],
                &[v.strides(), y_sub_shape.strides()],
                &[v.offset(), y_sub_shape.offset()],
            );

            let p = ctx.get_program(format!(
                r#"
                __kernel void concat(__global const {data_type} *x, __global {data_type} *y) {{
                    uint gid = get_global_id(0);
                    {idx_c}
                    y[idx_out] = x[idx_in];
                }}"#
            ));

            p.kernel("concat")
                .arg_tensor(v)
                .arg_tensor(&y)
                .global_work_size(v.size())
                .launch()
                .map_err(Error::Device)?;
        }
        Ok(y)
    }
}
