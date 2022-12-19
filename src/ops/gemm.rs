use crate::error::Error;
use crate::ops::core::{ceil, div_up};
use crate::ops::{Category, Compose, Composer};
use crate::session::context::{CachedAccess, Context};
use crate::shape::{Array, Shape};
use crate::tensor::{Tensor, TensorDesc};
use crate::var::{Fun, ToFun};
use smallvec::ToSmallVec;
use std::cmp::min;
use std::time::Instant;

#[derive(Clone, Debug)]
struct Gemm {
    input: [TensorDesc; 2],
    output: TensorDesc,
    transpose_a: bool,
    transpose_b: bool,
    cached: CachedAccess,
}

#[derive(Clone, Debug)]
struct GemmBatched {
    input: [TensorDesc; 2],
    output: TensorDesc,
    transpose_a: bool,
    transpose_b: bool,
    cached: CachedAccess,
}

pub fn matmul<V1, V2>(x1: V1, x2: V2) -> Fun
where
    V1: ToFun,
    V2: ToFun,
{
    let x1 = x1.to_fun();
    let x2 = x2.to_fun();

    //println!("{:?}", x1.extents());
    //println!("{:?}", x2.extents());

    let ext = [&x1.extents()[..x1.rank() - 1], &x2.extents()[1..]].concat();
    //println!("{:?}", &ext);

    let x1 = x1.view([x1.size() / x1.extent(-1), x1.extent(-1)]);
    let x2 = x2.view([x2.extent(0), x2.size() / x2.extent(0)]);

    //println!("{:?}", x1.extents());
    //println!("{:?}", x2.extents());

    gemm(x1, x2, false, false).view(ext)
}

pub fn matmul_batched<V1, V2>(x1: V1, x2: V2) -> Fun
where
    V1: ToFun,
    V2: ToFun,
{
    let x1 = x1.to_fun();
    let x2 = x2.to_fun();

    // find same dims
    let mut batch_ext = Array::new();
    for i in 0..(min(x1.rank() - 2, x2.rank() - 2)) {
        if x1.extent(i) == x2.extent(i) {
            batch_ext.push(x1.extent(i));
        } else {
            break;
        }
    }
    let batch_len = batch_ext.len();
    let batch_size = batch_ext.iter().product::<usize>();

    if batch_len == 0 {
        panic!("batch dim incorrect!");
    }

    //println!("{:?}", x1.extents());
    //println!("{:?}", x2.extents());

    let ext = [
        batch_ext.as_slice(),
        &x1.extents()[batch_len..(x1.rank() - 1)],
        &x2.extents()[(batch_len + 1)..],
    ]
    .concat();

    //println!("{:?}", &ext);

    let x1 = x1.view([
        batch_size,
        x1.size() / (x1.extent(-1) * batch_size),
        x1.extent(-1),
    ]);
    let x2 = x2.view([
        batch_size,
        x2.extent(batch_len),
        x2.size() / (x2.extent(batch_len) * batch_size),
    ]);

    //println!("{:?}", x1.extents());
    //println!("{:?}", x2.extents());

    gemm_batched(x1, x2, false, false).view(ext)
}

pub fn gemm<V1, V2>(x1: V1, x2: V2, t1: bool, t2: bool) -> Fun
where
    V1: ToFun,
    V2: ToFun,
{
    let x1 = x1.to_fun().organize();
    let x2 = x2.to_fun().organize();
    assert_eq!(x1.data_type(), x2.data_type());

    let x1_ext = x1.extents();
    let x2_ext = x2.extents();

    let (height, chan1, width, chan2) = match (t1, t2) {
        (false, false) => (x1_ext[0], x1_ext[1], x2_ext[1], x2_ext[0]),
        (false, true) => (x1_ext[0], x1_ext[1], x2_ext[0], x2_ext[1]),
        (true, false) => (x1_ext[1], x1_ext[0], x2_ext[1], x2_ext[0]),
        (true, true) => (x1_ext[1], x1_ext[0], x2_ext[0], x2_ext[1]),
    };
    assert_eq!(chan1, chan2);

    Fun::from_binary_op(
        Gemm {
            input: [x1.desc().clone(), x2.desc().clone()],
            output: TensorDesc {
                shape: Shape::new([height, width]),
                data_type: x1.data_type(),
            },
            transpose_a: t1,
            transpose_b: t2,
            cached: Default::default(),
        },
        x1,
        x2,
    )
}

impl Compose<2> for Gemm {
    fn input(&self) -> &[TensorDesc; 2] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, x: [&Fun; 2], _: &Fun, gy: &Fun) -> [Option<Fun>; 2] {
        let (gx1, gx2) = match (self.transpose_a, self.transpose_b) {
            (false, false) => (gemm(gy, x[1], false, true), gemm(x[0], gy, true, false)),
            (false, true) => (gemm(gy, x[1], false, false), gemm(gy, x[0], true, false)),
            (true, false) => (gemm(x[1], gy, false, true), gemm(x[0], gy, false, false)),
            (true, true) => (gemm(x[1], gy, true, true), gemm(gy, x[0], true, true)),
        };

        [Some(gx1), Some(gx2)]
    }

    fn compute(&self, x: [&Tensor; 2], ctx: &mut Context) -> Result<Tensor, Error> {
        self.compute_direct(x[0], x[1], ctx)
    }

    fn cat(&self) -> Category {
        Category::Contract
    }
}

impl Gemm {
    fn compute_indirect(&self, x1: &Tensor, x2: &Tensor, ctx: &mut Context) -> Tensor {
        todo!()
    }

    fn compute_direct(&self, x1: &Tensor, x2: &Tensor, ctx: &mut Context) -> Result<Tensor, Error> {
        let y = Tensor::uninit2(self.output(), ctx)?; // all defaults to gpu tensors

        y.fill_zero();
        let p = self.cached.load_program(ctx, || {
            format!(
                "{}\n{}",
                include_str!("kernels/common.opencl"),
                include_str!("kernels/matmul_direct.opencl")
            )
        });

        let use_direct = true;

        let (h_ax, c_ax, w_ax) = match (self.transpose_a, self.transpose_b) {
            (false, false) => (0, 1, 1),
            (false, true) => (0, 1, 0),
            (true, false) => (1, 0, 1),
            (true, true) => (1, 0, 0),
        };

        let a_tr = self.transpose_a != use_direct;
        let b_tr = self.transpose_b;
        let c_tr = use_direct;

        let prog_name = match (a_tr, b_tr) {
            (false, false) => "XgemmDirectNN",
            (false, true) => "XgemmDirectNT",
            (true, false) => "XgemmDirectTN",
            (true, true) => "XgemmDirectTT",
        };

        // x1: [m, k]
        // x2: [k, n]
        // y: [m, n]
        let m = x1.extent(h_ax);
        let k = x1.extent(c_ax);
        let n = x2.extent(w_ax);

        let a_ld = x1.strides()[0]; // [m, k]
        let b_ld = x2.strides()[0]; // [k, n]
        let c_ld = y.strides()[0]; // [m, n]

        let wgd = 8;
        let mdimcd = 8;
        let ndimcd = 8;

        let m_ceiled = ceil(m, wgd);
        let n_ceiled = ceil(n, wgd);

        p.kernel(prog_name)
            .arg(m as i32)
            .arg(n as i32)
            .arg(k as i32)
            .arg(1.0_f32)
            .arg(0.0_f32)
            .arg_tensor(x1)
            .arg(x1.offset() as i32)
            .arg(a_ld as i32)
            .arg_tensor(x2)
            .arg(x2.offset() as i32)
            .arg(b_ld as i32)
            .arg_tensor(&y)
            .arg(y.offset() as i32)
            .arg(c_ld as i32)
            .arg(c_tr as i32)
            .arg(0_i32)
            .arg(0_i32)
            .global_work_size([(m_ceiled * mdimcd) / wgd, (n_ceiled * ndimcd) / wgd])
            .local_work_size([mdimcd, ndimcd])
            .launch()
            .map_err(Error::Device)?;

        Ok(y)
    }
}

pub fn gemm_batched<V1, V2>(x1: V1, x2: V2, t1: bool, t2: bool) -> Fun
where
    V1: ToFun,
    V2: ToFun,
{
    let x1 = x1.to_fun().organize();
    let x2 = x2.to_fun().organize();

    assert_eq!(x1.extent(0), x2.extent(0));
    assert_eq!(x1.data_type(), x2.data_type());

    let x1_ext = &x1.extents()[1..];
    let x2_ext = &x2.extents()[1..];

    let batch = x1.extent(0);
    let (height, chan1, width, chan2) = match (t1, t2) {
        (false, false) => (x1_ext[0], x1_ext[1], x2_ext[1], x2_ext[0]),
        (false, true) => (x1_ext[0], x1_ext[1], x2_ext[0], x2_ext[1]),
        (true, false) => (x1_ext[1], x1_ext[0], x2_ext[1], x2_ext[0]),
        (true, true) => (x1_ext[1], x1_ext[0], x2_ext[0], x2_ext[1]),
    };
    assert_eq!(chan1, chan2);

    Fun::from_binary_op(
        GemmBatched {
            input: [x1.desc().clone(), x2.desc().clone()],
            output: TensorDesc {
                shape: Shape::new([batch, height, width]),
                data_type: x1.data_type(),
            },
            transpose_a: t1,
            transpose_b: t2,
            cached: Default::default(),
        },
        x1,
        x2,
    )
}

impl Compose<2> for GemmBatched {
    fn input(&self) -> &[TensorDesc; 2] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, x: [&Fun; 2], _: &Fun, gy: &Fun) -> [Option<Fun>; 2] {
        let (gx1, gx2) = match (self.transpose_a, self.transpose_b) {
            (false, false) => (
                gemm_batched(gy, x[1], false, true),
                gemm_batched(x[0], gy, true, false),
            ),
            (false, true) => (
                gemm_batched(gy, x[1], false, false),
                gemm_batched(gy, x[0], true, false),
            ),
            (true, false) => (
                gemm_batched(x[1], gy, false, true),
                gemm_batched(x[0], gy, false, false),
            ),
            (true, true) => (
                gemm_batched(x[1], gy, true, true),
                gemm_batched(gy, x[0], true, true),
            ),
        };

        [Some(gx1), Some(gx2)]
    }

    fn compute(&self, x: [&Tensor; 2], ctx: &mut Context) -> Result<Tensor, Error> {
        self.compute_direct(x[0], x[1], ctx)
    }

    fn cat(&self) -> Category {
        Category::Contract
    }
}

impl GemmBatched {
    fn compute_indirect(&self, x1: &Tensor, x2: &Tensor, ctx: &mut Context) -> Tensor {
        todo!()
    }

    fn compute_direct(&self, x1: &Tensor, x2: &Tensor, ctx: &mut Context) -> Result<Tensor, Error> {
        let y = Tensor::uninit2(self.output(), ctx)?; // all defaults to gpu tensors
        y.fill_zero();

        let p = self.cached.load_program(ctx, || {
            format!(
                "{}\n{}",
                include_str!("kernels/common.opencl"),
                include_str!("kernels/matmul_direct.opencl")
            )
        });

        let use_direct = true;

        let (h_ax, c_ax, w_ax) = match (self.transpose_a, self.transpose_b) {
            (false, false) => (0, 1, 1),
            (false, true) => (0, 1, 0),
            (true, false) => (1, 0, 1),
            (true, true) => (1, 0, 0),
        };

        let a_tr = self.transpose_a != use_direct;
        let b_tr = self.transpose_b;
        let c_tr = use_direct;

        let prog_name = match (a_tr, b_tr) {
            (false, false) => "XgemmDirectStridedBatchedNN",
            (false, true) => "XgemmDirectStridedBatchedNT",
            (true, false) => "XgemmDirectStridedBatchedTN",
            (true, true) => "XgemmDirectStridedBatchedTT",
        };

        // x1: [m, k]
        // x2: [k, n]
        // y: [m, n]

        let batch = x1.extent(0);
        let m = x1.extent(h_ax + 1);
        let k = x1.extent(c_ax + 1);
        let n = x2.extent(w_ax + 1);

        let a_ld = x1.strides()[1]; // [m, k]
        let b_ld = x2.strides()[1]; // [k, n]
        let c_ld = y.strides()[1]; // [m, n]

        let a_stride = x1.strides()[0];
        let b_stride = x2.strides()[0];
        let c_stride = y.strides()[0];

        let wgd = 8;
        let mdimcd = 8;
        let ndimcd = 8;

        let m_ceiled = ceil(m, wgd);
        let n_ceiled = ceil(n, wgd);

        p.kernel(prog_name)
            .arg(m as i32)
            .arg(n as i32)
            .arg(k as i32)
            .arg(1.0_f32)
            .arg(0.0_f32)
            .arg_tensor(x1)
            .arg(x1.offset() as i32)
            .arg(a_ld as i32)
            .arg(a_stride as i32)
            .arg_tensor(x2)
            .arg(x2.offset() as i32)
            .arg(b_ld as i32)
            .arg(b_stride as i32)
            .arg_tensor(&y)
            .arg(y.offset() as i32)
            .arg(c_ld as i32)
            .arg(c_stride as i32)
            .arg(c_tr as i32)
            .arg(0_i32)
            .arg(0_i32)
            .global_work_size([(m_ceiled * mdimcd) / wgd, (n_ceiled * ndimcd) / wgd, batch])
            .local_work_size([mdimcd, ndimcd])
            .launch()
            .map_err(Error::Device)?;

        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::gemm::gemm;
    use crate::session::context::Context;
    use crate::tensor::Tensor;
    use crate::var::{grad_check, Fun};

    #[test]
    fn test_gemm() {
        let mut ctx = Context::new();

        // rectangle gemm
        // (7, 13) * (13, 5) -> (7, 5)

        let a = Tensor::new([
            [
                -8.6056e-01,
                -2.9842e-01,
                1.9980e+00,
                -4.5024e-01,
                -2.4271e-01,
                5.2440e-01,
                8.8012e-01,
                -4.1342e-01,
                2.2987e-03,
                9.8891e-01,
                8.8611e-01,
                5.6608e-01,
                -2.0152e+00,
            ],
            [
                9.4158e-01,
                6.9302e-02,
                5.4060e-01,
                7.6085e-02,
                -3.5421e-01,
                1.6967e+00,
                1.0691e+00,
                -2.8253e-01,
                2.4041e-01,
                5.8997e-01,
                1.5264e+00,
                -7.6517e-01,
                -5.7514e-01,
            ],
            [
                -3.7834e-01,
                4.8413e-01,
                -9.0677e-01,
                4.6206e-01,
                -1.8329e-01,
                1.7960e+00,
                -1.3886e-01,
                -2.2288e+00,
                4.2476e-01,
                3.7146e-01,
                1.4474e-01,
                1.1426e+00,
                -1.8239e+00,
            ],
            [
                3.7081e-01,
                1.6123e-01,
                -9.7269e-01,
                3.7174e-01,
                7.2450e-01,
                -4.1521e-01,
                1.4622e+00,
                -1.1035e+00,
                2.1761e-01,
                2.4145e+00,
                -1.0204e+00,
                -1.2136e+00,
                1.7628e+00,
            ],
            [
                -2.3934e-01,
                5.1756e-01,
                -1.0689e+00,
                4.4095e-01,
                -1.1806e+00,
                -1.1398e-01,
                8.6690e-01,
                2.7034e-01,
                1.1642e+00,
                1.5112e+00,
                -8.9659e-01,
                4.2487e-01,
                6.4119e-01,
            ],
            [
                -8.5414e-01,
                2.7088e+00,
                -1.1474e-01,
                -4.6268e-01,
                -1.2554e+00,
                1.3025e-01,
                -1.3815e-01,
                1.0025e+00,
                -4.8001e-01,
                -1.6552e+00,
                -7.7931e-02,
                2.0460e+00,
                -2.1500e+00,
            ],
            [
                3.4902e-01,
                -2.4865e-01,
                -2.8962e-01,
                6.8369e-01,
                -5.6865e-01,
                -1.3244e+00,
                6.9352e-01,
                1.5472e+00,
                -4.2794e-02,
                -1.6029e+00,
                -1.3639e-01,
                -1.1080e+00,
                -6.4896e-01,
            ],
        ])
        .to_device(&mut ctx);

        let b = Tensor::new([
            [0.1906, -0.6252, 2.7072, -1.2903, 0.1752],
            [-0.3901, -0.3498, 0.2213, 0.4817, 0.4615],
            [0.3100, 0.3822, -0.3150, -0.8875, -0.9193],
            [0.1669, -0.2752, 1.6298, -0.8154, -1.8860],
            [-1.1700, -0.0616, -0.4734, -1.2094, 1.3512],
            [0.4448, -0.1040, 1.7809, 1.2843, -3.3656],
            [-0.0631, 0.6526, 1.8588, 0.5167, 0.0903],
            [-0.6347, 0.6246, -0.6222, -0.6172, 0.3701],
            [-0.8286, -0.0523, 1.3460, 0.7634, 0.8137],
            [-0.1849, -0.6256, 2.0959, 0.7965, 1.3486],
            [1.2730, -1.8608, 0.3819, 0.5514, 0.0603],
            [0.7427, -0.7315, 0.3690, 0.1338, 1.6253],
            [0.9719, 0.8329, 0.2302, -1.1798, -0.9891],
        ])
        .to_device(&mut ctx);

        let c_gt = Tensor::new([
            [0.6257, -2.5535, 1.3423, 4.9689, 0.8597],
            [2.1212, -3.2018, 9.5986, 3.6821, -6.2319],
            [0.8112, -4.9437, 5.9428, 7.9864, -2.4939],
            [-1.7697, 2.2340, 9.1959, -0.5290, 2.0881],
            [-0.8477, 1.1432, 7.0176, 3.6082, 2.3021],
            [-0.3987, -1.8119, -6.3702, 4.9641, 3.1271],
            [-2.0571, 2.6904, -3.6950, -3.2476, -0.1162],
        ])
        .to_device(&mut ctx);

        //let a = Tensor::ones([5, 5]).to_device(&ctx);
        //let b = Tensor::ones([5, 5]).to_device(&ctx);
        let a = Fun::new(a);
        let b = Fun::new(b);
        let c = gemm(&a, &b, false, false);
        assert!(Tensor::all_close(&c.eval(&mut ctx), &c_gt, 0.001));

        assert!(grad_check(&c, &a, 0.05, &mut ctx));
        assert!(grad_check(&c, &b, 0.05, &mut ctx));

        // square gemm + transpose

        let a = Tensor::new([
            [-3.1245, 0.2204, 1.3917, -0.7569, -0.8674, 0.8345, 1.1989],
            [-0.6063, -1.9395, -0.4651, 0.5267, -1.6963, -2.3543, -0.8237],
            [0.1656, -1.7071, -0.5037, -2.1910, 0.7057, -0.7724, -0.4315],
            [-1.6404, -1.4849, -0.6444, 0.6801, 0.3912, 0.5068, 0.6383],
            [-0.6541, -0.3181, -1.6829, -0.0261, -0.7308, 0.4353, -0.5598],
            [-1.6514, 0.8159, 0.7137, -1.7617, 1.0759, 0.7614, 1.8485],
            [-0.7207, -0.8405, 0.7565, 0.0713, 1.4233, 0.1446, -1.7953],
        ])
        .to_device(&mut ctx);

        let b = Tensor::new([
            [
                -3.3736e-01,
                -1.5805e+00,
                -3.9470e-01,
                7.5618e-01,
                7.6135e-01,
                1.1996e-01,
                -7.8769e-01,
            ],
            [
                7.1667e-01,
                -7.2413e-01,
                1.0056e+00,
                1.0400e+00,
                -3.2988e-01,
                1.7930e-01,
                2.0654e+00,
            ],
            [
                1.2656e+00,
                -1.9489e-01,
                4.5171e-01,
                -8.0885e-02,
                1.6077e+00,
                1.3891e+00,
                -1.5486e+00,
            ],
            [
                4.5950e-01, 1.1993e-01, 2.2889e-01, 4.8202e-01, 1.2650e+00, 1.8347e+00, 9.9623e-04,
            ],
            [
                -1.7365e+00,
                3.0852e-01,
                5.4687e-02,
                4.3470e-01,
                -7.0341e-01,
                7.7887e-01,
                -6.1995e-01,
            ],
            [
                -7.9349e-02,
                -3.1646e-01,
                -1.6119e+00,
                -4.6297e-01,
                1.9138e+00,
                -2.7651e-02,
                5.2975e-01,
            ],
            [
                1.2242e+00,
                -4.1216e-01,
                -2.4856e-01,
                -4.1229e-01,
                -1.9540e+00,
                2.5483e-01,
                -1.5341e+00,
            ],
        ])
        .to_device(&mut ctx);

        let cff_gt = Tensor::new([
            [5.5333, 3.3907, 0.2196, -3.8686, -1.3069, -0.1841, -0.0990],
            [0.5920, 3.0777, 2.1065, -1.4918, -1.6064, -1.5662, -1.7394],
            [-4.6160, 1.4498, -1.1201, -1.8233, -4.0235, -4.5449, -3.0630],
            [-0.9519, 3.5722, -1.9352, -2.7325, -1.4873, 0.3432, -1.7296],
            [-1.6000, 1.4565, -1.4304, -0.9903, -0.6906, -3.2449, 4.0067],
            [1.5697, 0.9979, -0.2366, -1.9540, -5.5189, -1.0045, -1.2206],
            [-4.0500, 2.7421, 0.0883, -0.1540, 3.8183, 1.5915, -0.3913],
        ]);

        let ctf_gt = Tensor::new([
            [0.4600, 5.7660, 3.1281, -3.0200, -5.2796, -3.9106, 1.5872],
            [-4.8483, 1.2009, -4.2718, -2.5975, -0.3876, -5.9015, 0.3816],
            [2.0554, -2.8987, -2.8226, -1.0751, 0.6597, -2.9361, -1.0167],
            [-1.5552, 1.8436, 2.8149, 1.2553, -6.9044, -1.7455, 4.0514],
            [3.0759, 1.3561, -3.0831, -3.6912, 1.3205, 1.0537, -5.0731],
            [-3.3528, 0.4310, -4.1692, -1.7336, 1.6798, -0.1103, -4.4117],
            [-2.6199, -1.1554, -3.9142, 0.0336, 8.7375, -0.3767, 2.1038],
        ]);

        let cft_gt = Tensor::new([
            [-1.9206, 1.1253, -5.3992, -1.0205, 5.7577, -2.7628, -3.8812],
            [2.9266, -0.5137, -5.3640, -6.8299, 0.5281, -2.4498, 3.9338],
            [1.9687, -2.6929, 1.2218, -1.8249, -2.6247, 3.4967, 1.0212],
            [3.5246, 1.2392, -1.7883, 0.6738, 2.3746, 2.3967, -3.1311],
            [1.3048, -2.7949, -1.2271, -0.8627, 2.1345, 1.1702, 2.1571],
            [-2.8917, 0.7106, -1.8591, 1.4131, 1.0829, 2.5553, -6.5531],
            [3.8419, -3.2246, 4.8569, 1.8395, 1.2890, 0.8397, -0.7435],
        ]);

        let ctt_gt = Tensor::new([
            [0.5781, -4.9085, -5.8580, -6.1191, 4.1554, -0.6556, -0.9770],
            [3.0596, -3.1832, 1.9293, -0.1442, -0.3396, 2.9588, 4.2247],
            [-1.8143, 2.4031, -1.2090, -0.6607, -1.5972, -1.6926, 4.5960],
            [0.5147, -2.5799, -4.7047, -3.7234, 0.2549, 3.1468, -1.3868],
            [1.4424, 5.0969, -2.3645, 0.7992, 1.6613, -1.3873, -1.1810],
            [4.4364, 2.3449, 2.6587, 2.1164, -1.8003, 2.5779, 1.0967],
            [2.7600, -1.5062, 5.8790, 3.3425, 0.8644, -1.5079, 5.9703],
        ]);

        let c = gemm(&a, &b, false, false).eval(&mut ctx);
        assert!(Tensor::all_close(&c, &cff_gt, 0.001));

        let c = gemm(&a, &b, true, false).eval(&mut ctx);
        assert!(Tensor::all_close(&c, &ctf_gt, 0.001));

        let c = gemm(&a, &b, false, true).eval(&mut ctx);
        assert!(Tensor::all_close(&c, &cft_gt, 0.001));

        let c = gemm(&a, &b, true, true).eval(&mut ctx);
        assert!(Tensor::all_close(&c, &ctt_gt, 0.001));

        let a = Fun::new(a);
        let b = Fun::new(b);

        // gemm grads

        let c = gemm(&a, &b, false, false);
        assert!(grad_check(&c, &a, 0.01, &mut ctx));
        assert!(grad_check(&c, &b, 0.01, &mut ctx));

        let c = gemm(&a, &b, false, true);
        assert!(grad_check(&c, &a, 0.01, &mut ctx));
        assert!(grad_check(&c, &b, 0.01, &mut ctx));

        let c = gemm(&a, &b, true, false);
        assert!(grad_check(&c, &a, 0.01, &mut ctx));
        assert!(grad_check(&c, &b, 0.01, &mut ctx));

        let c = gemm(&a, &b, true, true);
        assert!(grad_check(&c, &a, 0.01, &mut ctx));
        assert!(grad_check(&c, &b, 0.01, &mut ctx));
    }
}
