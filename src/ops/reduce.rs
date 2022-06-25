use crate::error::Error;
use crate::ops::core::translate_id;
use crate::ops::map::{cond, eq, scalar};
use crate::ops::{Category, NaryOperator, Operator};
use crate::session::context::{CachedAccess, Context};
use crate::shape::{display_comma, Array, Axes, Axis, Shape};
use crate::tensor::data::DataType;
use crate::tensor::{Tensor, TensorDesc};
use crate::var::{Var, Variable};
use itertools::Itertools;

// refer_id, refer_stride, [target_ids],  [target_strides]
// tensors -> string

#[derive(Copy, Clone, Debug)]
pub enum Reduce {
    Sum,
    Prod,
    Max,
    Min,
}

#[derive(Copy, Clone, Debug)]
pub enum ReduceArg {
    Max,
    Min,
}

#[derive(Clone, Debug)]
pub struct ReduceOperator {
    input: [TensorDesc; 1],
    output: TensorDesc,

    reduce: Reduce,
    axes: Array,
    preserve_axes: bool,
    cached: CachedAccess,
}

#[derive(Clone, Debug)]
pub struct ReduceArgOperator {
    input: [TensorDesc; 1],
    output: TensorDesc,

    reduce: ReduceArg,
    axis: usize,
    preserve_axes: bool,
    cached: CachedAccess,
}

pub fn sum<V, A>(x: V, axes: A, preserve_axes: bool) -> Var
where
    V: Variable,
    A: Axes,
{
    reduce(x, axes, preserve_axes, Reduce::Sum)
}

pub fn mean<V, A>(x: V, axes: A, preserve_axes: bool) -> Var
where
    V: Variable,
    A: Axes,
{
    let x = x.into_var();
    let axes = axes.to_arr(x.rank()).unwrap();
    let size = axes.iter().map(|axis| x.extent(*axis)).product::<usize>() as f32;

    let r = reduce(x, axes, preserve_axes, Reduce::Sum);

    &r / scalar(size, r.extents())
}

pub fn var<V, A>(x: V, axes: A, preserve_axes: bool) -> Var
where
    V: Variable,
    A: Axes,
{
    let x = x.into_var();
    let axes = axes.to_arr(x.rank()).unwrap();

    x.square().mean(&axes, preserve_axes) - x.mean(&axes, preserve_axes).square()
}

pub fn prod<V, A>(x: V, axes: A, preserve_axes: bool) -> Var
where
    V: Variable,
    A: Axes,
{
    reduce(x, axes, preserve_axes, Reduce::Prod)
}

pub fn max<V, A>(x: V, axes: A, preserve_axes: bool) -> Var
where
    V: Variable,
    A: Axes,
{
    reduce(x, axes, preserve_axes, Reduce::Max)
}

pub fn min<V, A>(x: V, axes: A, preserve_axes: bool) -> Var
where
    V: Variable,
    A: Axes,
{
    reduce(x, axes, preserve_axes, Reduce::Min)
}

pub fn argmax<V, A>(x: V, axis: A, preserve_axes: bool) -> Var
where
    V: Variable,
    A: Axis,
{
    reduce_arg(x, axis, preserve_axes, ReduceArg::Max)
}

pub fn argmin<V, A>(x: V, axis: A, preserve_axes: bool) -> Var
where
    V: Variable,
    A: Axis,
{
    reduce_arg(x, axis, preserve_axes, ReduceArg::Min)
}

fn reduced_extents(extents: &[usize], axes: &[usize], preserve_axes: bool) -> Array {
    let mut out = Array::new();

    for i in 0..extents.len() {
        if !axes.contains(&i) {
            out.push(extents[i]);
        } else if preserve_axes {
            out.push(1);
        }
    }

    if out.is_empty() {
        out.push(1);
    }

    out
}

fn compute_reduce_arg(
    x: &Tensor,
    ctx: &mut Context,
    axis: usize,
    preserve_axes: bool,
    reduce_arg_fn: fn(&str, &str, &str, &str) -> String,
) -> Result<Tensor, Error> {
    let extent = reduced_extents(x.extents(), &[axis], preserve_axes);

    let (inner, outer) = x.shape().extract(&[axis], true);

    let y = Tensor::uninit(extent, DataType::Uint, ctx)?; // all defaults to gpu tensors

    let inner_stride = inner.strides()[0];
    let inner_extent = inner.extents()[0];

    let data_type = x.data_type().opencl();

    let p = ctx.get_program(format!(
        r#"
        __kernel void reduce_arg(__global const {data_type} *x, __global uint *y) {{

            uint gid = get_global_id(0);
            
            {outer_idx_c}

            uint inner_strides = {inner_stride};
            
            {data_type} sum = x[outer_idx];
            uint arg_idx = 0;
            uint inner_idx = 0;

            for (uint i = 0; i < {inner_extent}; ++i) {{
               inner_idx = i * inner_strides;
               {red}
            }}

            y[gid] = arg_idx;
        }}
        "#,
        outer_idx_c = translate_id(
            "gid",
            &Shape::default_strides(outer.extents()),
            &["outer_idx"],
            &[outer.strides()],
            &[outer.offset()],
        )
        .0,
        red = reduce_arg_fn("sum", "arg_idx", "x[outer_idx + inner_idx]", "i")
    ));

    p.kernel("reduce_arg")
        .arg_tensor(x)
        .arg_tensor(&y)
        .global_work_size(y.size())
        .launch()
        .map_err(|e| Error::Device(e))?;

    Ok(y)
}

pub fn reduce<V, A>(x: V, axes: A, preserve_axes: bool, op: Reduce) -> Var
where
    V: Variable,
    A: Axes,
{
    let x = x.into_var();
    let axes = axes.to_arr(x.rank()).unwrap();

    Var::from_unary_op(
        ReduceOperator {
            input: [x.desc().clone()],
            output: TensorDesc::new(
                reduced_extents(x.extents(), &axes, preserve_axes),
                x.data_type(),
            ),
            reduce: op,
            axes,
            preserve_axes,
            cached: Default::default(),
        },
        x,
    )
}

impl NaryOperator<1> for ReduceOperator {
    fn input(&self) -> &[TensorDesc; 1] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, x: [&Var; 1], y: &Var, gy: &Var) -> [Option<Var>; 1] {
        // println!("-------------------");
        // println!("x {:?}", x.shape());
        // println!("y {:?}", y.shape());
        // println!("gy {:?}", gy.shape());
        let x = x[0];
        let (y, gy) = if self.preserve_axes {
            (y.clone(), gy.clone())
        } else {
            let extents = reduced_extents(self.input[0].extents(), &self.axes, true);

            (y.view(&extents), gy.view(&extents))
        };

        [match self.reduce {
            Reduce::Sum => Some(gy.expand(x.extents())),
            Reduce::Prod => Some(gy * y / x),
            Reduce::Max => Some(cond(eq(x, y), gy, 0.0)),
            Reduce::Min => Some(cond(eq(x, y), gy, 0.0)),
        }]
    }

    fn compute(&self, x: [&Tensor; 1], ctx: &mut Context) -> Result<Tensor, Error> {
        let x = x[0];
        let y = Tensor::uninit2(self.output(), ctx)?; // all defaults to gpu tensors

        self.cached
            .load_program(ctx, || {
                let reduce_fn: fn(&str, &str) -> String = match self.reduce {
                    Reduce::Sum => |acc, val| format!("{acc} = {acc} + {val};"),
                    Reduce::Prod => |acc, val| format!("{acc} = {acc} * {val};"),
                    Reduce::Max => |acc, val| format!("{acc} = max({acc}, {val});"),
                    Reduce::Min => |acc, val| format!("{acc} = min({acc}, {val});"),
                };

                let (inner, outer) = x.shape().extract(&self.axes, true);
                let inner_strides = display_comma(inner.strides());
                let inner_ndim = inner.num_axes();
                let data_type = x.data_type().opencl();

                format!(
                    r#"
            __kernel void reduce(__global const {data_type} *x, __global {data_type} *y) {{
                uint gid = get_global_id(0);
                {outer_idx_c}
                uint inner_strides[{inner_ndim}] = {{{inner_strides}}};
                {data_type} sum = 0;
                uint inner_idx = 0;
                {inner_loop_c}
                y[gid] = sum;
            }}
            "#,
                    outer_idx_c = translate_id(
                        "gid",
                        &Shape::default_strides(outer.extents()),
                        &["outer_idx"],
                        &[outer.strides()],
                        &[outer.offset()],
                    )
                    .0,
                    inner_loop_c = (0..inner_ndim).fold(
                        format!(
                            r#"
                    inner_idx = {csum};
                    {red}
                    "#,
                            red = reduce_fn("sum", "x[outer_idx + inner_idx]"),
                            csum = (0..inner_ndim).map(|n| format!("c{n}")).join("+")
                        ),
                        |acc, n| {
                            format!(
                                r#"
                        for (uint i{n} = 0; i{n} < {e}; ++i{n}) {{
                            uint c{n} = i{n} * inner_strides[{n}];
                            {acc}
                        }}"#,
                                e = inner.extents()[n]
                            )
                        },
                    )
                )
            })
            .kernel("reduce")
            .arg_tensor(x)
            .arg_tensor(&y)
            .global_work_size(y.size())
            .launch()
            .map_err(|e| Error::Device(e))?;
        Ok(y)
    }

    fn cat(&self) -> Category {
        Category::Reduce(self.clone())
    }
}

pub fn reduce_arg<V, A>(x: V, axis: A, preserve_axes: bool, op: ReduceArg) -> Var
where
    V: Variable,
    A: Axis,
{
    let x = x.into_var();
    let axis = axis.to_usize(x.rank()).unwrap();

    Var::from_unary_op(
        ReduceArgOperator {
            input: [x.desc().clone()],
            output: TensorDesc::new(
                reduced_extents(x.extents(), &[axis], preserve_axes),
                DataType::Uint,
            ),
            reduce: op,
            axis,
            preserve_axes,
            cached: Default::default(),
        },
        x,
    )
}

impl NaryOperator<1> for ReduceArgOperator {
    fn input(&self) -> &[TensorDesc; 1] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, _: [&Var; 1], _: &Var, _: &Var) -> [Option<Var>; 1] {
        [match self.reduce {
            ReduceArg::Max => None,
            ReduceArg::Min => None,
        }]
    }

    fn compute(&self, x: [&Tensor; 1], ctx: &mut Context) -> Result<Tensor, Error> {
        let x = x[0];
        let y = Tensor::uninit2(self.output(), ctx)?; // all defaults to gpu tensors

        self.cached
            .load_program(ctx, || {
                let reduce_arg_fn: fn(&str, &str, &str, &str) -> String = match self.reduce {
                    ReduceArg::Max => |acc, acc_idx, val, val_idx| {
                        format!("if ({acc} < {val}) {{ {acc_idx} = {val_idx}; {acc} = {val}; }}")
                    },
                    ReduceArg::Min => |acc, acc_idx, val, val_idx| {
                        format!("if ({acc} > {val}) {{ {acc_idx} = {val_idx}; {acc} = {val}; }}")
                    },
                };

                let extent = reduced_extents(x.extents(), &[self.axis], self.preserve_axes);

                let (inner, outer) = x.shape().extract(&[self.axis], true);

                let inner_stride = inner.strides()[0];
                let inner_extent = inner.extents()[0];

                let data_type = x.data_type().opencl();

                format!(
                    r#"
            __kernel void reduce_arg(__global const {data_type} *x, __global uint *y) {{
                uint gid = get_global_id(0);
                {outer_idx_c}
                uint inner_strides = {inner_stride};
                {data_type} sum = x[outer_idx];
                uint arg_idx = 0;
                uint inner_idx = 0;
                for (uint i = 0; i < {inner_extent}; ++i) {{
                   inner_idx = i * inner_strides;
                   {red}
                }}
                y[gid] = arg_idx;
            }}
            "#,
                    outer_idx_c = translate_id(
                        "gid",
                        &Shape::default_strides(outer.extents()),
                        &["outer_idx"],
                        &[outer.strides()],
                        &[outer.offset()],
                    )
                    .0,
                    red = reduce_arg_fn("sum", "arg_idx", "x[outer_idx + inner_idx]", "i")
                )
            })
            .kernel("reduce_arg")
            .arg_tensor(x)
            .arg_tensor(&y)
            .global_work_size(y.size())
            .launch()
            .map_err(|e| Error::Device(e))?;

        Ok(y)
    }

    fn cat(&self) -> Category {
        Category::Other
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::reduce::{argmax, max, mean, min, prod};
    use crate::session::context::Context;
    use crate::tensor::Tensor;
    use crate::var::{grad_check, Var};

    #[test]
    fn test_reduce() {
        let mut ctx = Context::new();

        let x = Tensor::new([
            [
                [
                    [[1.0080, 0.1239], [0.8520, -1.4361], [1.4414, -0.4076]],
                    [[-0.6060, 0.6664], [0.9516, -0.5973], [0.2697, -0.1846]],
                ],
                [
                    [[-0.5995, 1.5344], [0.7107, -0.5441], [1.0940, 1.5698]],
                    [[-0.5274, 1.6633], [0.3242, -0.1302], [-1.9938, -0.2440]],
                ],
                [
                    [[-0.4944, 0.5539], [0.1082, 0.0160], [-1.8767, -0.2652]],
                    [[-1.2379, 0.7148], [0.1939, -0.2506], [0.4936, 0.3340]],
                ],
            ],
            [
                [
                    [[0.3243, -1.0333], [-0.3338, 2.4103], [0.1622, 0.5120]],
                    [[-1.0259, -0.9454], [0.0027, 0.2757], [0.9749, -1.7382]],
                ],
                [
                    [[-0.0894, -0.7508], [-0.5758, -0.9779], [-0.0558, 0.9385]],
                    [[-0.3390, -0.0886], [1.3336, -1.0862], [0.8329, -0.5927]],
                ],
                [
                    [[-1.5139, 1.0410], [0.2499, -0.8632], [-1.8819, 1.3997]],
                    [[0.2809, -1.1449], [-0.4356, 1.2388], [-0.1237, -0.9235]],
                ],
            ],
        ])
        .to_device(&mut ctx);

        let x = Var::new(x);
        let y_gt = Tensor::new([[0.3351, -0.1962], [0.0096, 0.1076], [-0.5198, 0.1542]]);

        let y = mean(&x, [0, 2, 3], false);

        assert!(Tensor::all_close(&y.eval(&mut ctx), &y_gt, 0.001));

        assert!(grad_check(&y, &x, 0.1, &mut ctx));

        let y = prod(&x, [0, 2, 3], false);
        assert!(grad_check(&y, &x, 0.1, &mut ctx));
        //
        // let y = max(&x, [0], true);
        // assert!(grad_check(&y, &x, 0.01, &mut ctx));
        //
        // let y = min(&x, [0, 2, 3], false);
        // assert!(grad_check(&y, &x, 0.1, &mut ctx));
    }

    #[test]
    fn test_reduce_arg() {
        let mut ctx = Context::new();

        let x = Tensor::new([
            [
                [-0.3532, 0.9828, -0.8207, 1.1685],
                [-0.0931, 0.5805, -0.0843, 0.0327],
                [-1.2298, -0.8452, 0.6602, 1.0798],
            ],
            [
                [0.3521, -0.0408, -0.3169, 0.4699],
                [-0.3661, -1.5492, 0.8688, 0.1848],
                [1.0493, 0.6149, -1.0591, 0.7036],
            ],
        ])
        .to_device(&mut ctx);

        let y_gt = Tensor::new([[1_u32, 0, 2, 0], [2, 2, 1, 2]]);

        let y = argmax(x, 1, false).eval(&mut ctx);

        assert!(Tensor::all_equal::<u32>(&y, &y_gt));
    }
}
