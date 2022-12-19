use crate::ops::map::{ge, gt, sign};
use crate::shape::{Axes, Extent};
use crate::var::{Fun, ToFun};

// nn

struct Resize;

struct Dropout;

struct Einsum;

// activations

struct Sigmoid;

struct Relu;

struct LeakyRelu;

struct Softmax;

struct LogSoftmax;

// loss

struct MeanSquaredError;

struct SoftmaxCrossEntropy;

struct BinaryCrossEntropy;

pub fn relu<V>(x: V) -> Fun
where
    V: ToFun,
{
    let x = x.to_fun();

    gt(&x, 0.0).float() * x
}

pub fn softmax<V, A>(x: V, axes: A) -> Fun
where
    V: ToFun,
    A: Axes,
{
    let x = x.to_fun();
    let axes = axes.to_arr(x.rank()).unwrap();

    // For numerical stability
    let xm = x.max(&axes, true);
    let y = (x - xm).exp();
    let sum = y.sum(&axes, true);

    y / sum
}

pub fn log_sum_exp<V, A>(x: V, axes: A) -> Fun
where
    V: ToFun,
    A: Axes,
{
    let x = x.to_fun();
    let axes = axes.to_arr(x.rank()).unwrap();

    let c = x.max(&axes, true);
    (x - &c).exp().sum(&axes, true).log() + c
}

pub fn softmax_cross_entropy<V1, V2>(x1: V1, x2: V2) -> Fun
where
    V1: ToFun,
    V2: ToFun,
{
    let x1 = x1.to_fun();
    let x2 = x2.to_fun();

    let log_z = &x1 - log_sum_exp(&x1, 1);
    let log_p = log_z.gather(x2, 1); //log_z * x2;

    -log_p.sum(1, false)
}

pub fn layer_norm<V1, V2, V3, A>(x: V1, axes: A, gamma: V2, beta: V3, eps: f32) -> Fun
where
    V1: ToFun,
    V2: ToFun,
    V3: ToFun,
    A: Axes,
{
    let x = x.to_fun();
    let axes = axes.to_arr(x.rank()).unwrap();

    let mean = x.mean(&axes, true);
    let var = x.var(&axes, true);

    let xc = (x - mean) / (var + eps).sqrt();

    xc * gamma + beta
}

#[cfg(test)]
mod tests {
    use crate::ops::conv::max_pool_2d;
    use crate::ops::nn::{relu, softmax, softmax_cross_entropy};
    use crate::session::context::Context;
    use crate::tensor::Tensor;
    use crate::var::{grad_check, Fun};

    #[test]
    fn test_relu() {
        let mut ctx = Context::new();

        let x = Tensor::new([
            [0.3199, 0.1686, 0.9691],
            [0.0165, 1.7377, -0.1908],
            [0.0494, -0.4472, 0.5780],
        ])
        .to_device(&mut ctx);

        let y_gt = Tensor::new([
            [0.3199, 0.1686, 0.9691],
            [0.0165, 1.7377, 0.0000],
            [0.0494, 0.0000, 0.5780],
        ]);
        let x = Fun::new(x);
        let y = relu(&x);
        assert!(Tensor::all_close(&y.eval(&mut ctx), &y_gt, 0.001));
        assert!(grad_check(&y, &x, 0.01, &mut ctx));
    }

    #[test]
    fn test_softmax() {
        let mut ctx = Context::new();

        let x = Tensor::new([
            [1.4971, -0.0897, -0.8237],
            [-0.2428, -0.5554, -1.3207],
            [-2.0004, -0.8406, -0.9005],
        ])
        .to_device(&mut ctx);

        let y_gt = Tensor::new([
            [0.7676, 0.1570, 0.0754],
            [0.4826, 0.3531, 0.1643],
            [0.1390, 0.4434, 0.4176],
        ])
        .to_device(&mut ctx);

        let x = Fun::new(x);
        let y = softmax(&x, 1);
        assert!(Tensor::all_close(&y.eval(&mut ctx), &y_gt, 0.001));
        assert!(grad_check(&y, &x, 0.01, &mut ctx));
    }

    #[test]
    fn test_softmax_cross_entropy() {
        let mut ctx = Context::new();

        let x = Tensor::new([
            [
                0.0681, -0.4750, -0.1068, 0.2453, -0.5245, 0.1971, 0.0826, -0.4771, 0.7162, -1.5326,
            ],
            [
                -2.1222, 2.6529, 0.1163, 2.4620, -0.3893, -0.7439, -0.1908, -0.2767, 1.4722, 0.2627,
            ],
            [
                0.7419, 0.3707, 0.0854, 0.3992, -2.4740, -0.9155, -0.7988, 0.1836, -0.3489, 0.1029,
            ],
            [
                -0.4769, 0.6530, 0.8418, 0.6481, 0.1508, 0.9778, 2.2582, 0.8823, -0.2821, 1.3810,
            ],
            [
                -0.4457, 2.3899, 0.3116, 1.1650, 0.4207, 1.6690, -1.9891, -0.2580, 0.6080, -1.3612,
            ],
        ])
        .to_device(&mut ctx);

        let label = Tensor::new([[0], [1], [2], [3], [4]]).to_device(&mut ctx);

        let gx_gt = Tensor::new([
            [
                -0.1778, 0.0129, 0.0186, 0.0265, 0.0123, 0.0252, 0.0225, 0.0129, 0.0424, 0.0045,
            ],
            [
                0.0007, -0.1202, 0.0063, 0.0660, 0.0038, 0.0027, 0.0046, 0.0043, 0.0245, 0.0073,
            ],
            [
                0.0417, 0.0287, -0.1784, 0.0296, 0.0017, 0.0079, 0.0089, 0.0238, 0.0140, 0.0220,
            ],
            [
                0.0045, 0.0141, 0.0170, -0.1860, 0.0085, 0.0195, 0.0701, 0.0177, 0.0055, 0.0291,
            ],
            [
                0.0049, 0.0841, 0.0105, 0.0247, -0.1883, 0.0409, 0.0011, 0.0060, 0.0142, 0.0020,
            ],
        ]);

        let x = Fun::new(x);

        let loss_gt = Tensor::new([2.1987, 0.9184, 2.2250, 2.6592, 2.8357]).to_device(&mut ctx);

        let loss = softmax_cross_entropy(&x, &label);

        // forward check
        assert!(Tensor::all_close(&loss.eval(&mut ctx), &loss_gt, 0.001));
        assert!(grad_check(&loss, &x, 0.01, &mut ctx));

        // backward check

        //
        // // forward check
        // assert!(loss.data().equals(&loss_data, 0.001));
        //
        // let grads = diff(&loss, &[&input]);
        //
        // let input_grad = grads.get(&input).unwrap();
        //
        // // backward check
        // assert!(input_grad.data().equals(&gx, 0.001));
    }

    #[test]
    fn test_max_pool_2d() {
        let mut ctx = Context::new();
        let x = Tensor::new([
            [
                [
                    [0.1870, -0.5734, -0.4814],
                    [0.7338, -2.5648, 1.4118],
                    [-0.8794, -0.5410, 0.3624],
                    [-1.6200, -0.0506, -1.2508],
                    [-1.0860, 0.2241, -0.2400],
                ],
                [
                    [-1.2149, 1.1953, 0.7561],
                    [1.7142, -0.4472, -0.3679],
                    [-1.0746, 0.3479, -0.5167],
                    [-1.3591, 1.8230, -0.7538],
                    [0.4459, 0.0873, 0.0470],
                ],
                [
                    [1.3058, 0.0904, -0.0755],
                    [0.2551, 0.7054, -0.4256],
                    [-0.4351, -0.4127, 0.1646],
                    [0.8096, 1.4073, -0.9770],
                    [-1.0261, 0.2592, -0.3753],
                ],
                [
                    [0.8594, -0.0220, -0.0413],
                    [1.8615, 0.3437, -1.9272],
                    [-0.3062, 1.0650, 0.4055],
                    [-0.8764, -0.8571, -0.4017],
                    [0.2141, 0.5499, -0.0531],
                ],
                [
                    [-1.2101, -1.5427, 1.4202],
                    [1.3626, 2.1217, 0.0761],
                    [-0.2612, 0.9447, -0.3324],
                    [-1.9307, 1.2795, -0.2247],
                    [0.5893, -1.3706, 0.9963],
                ],
            ],
            [
                [
                    [-0.5255, -0.2076, -1.0913],
                    [1.8998, 0.9345, 0.1250],
                    [0.7396, 0.1725, 0.2215],
                    [-0.0420, 0.9993, 0.0482],
                    [-1.0233, -0.3075, 0.7745],
                ],
                [
                    [-1.0745, 0.6724, 1.3045],
                    [1.0410, 0.7017, 1.2738],
                    [0.2491, 0.5413, -0.6103],
                    [0.9705, 2.0919, 0.4034],
                    [-0.2245, -0.0532, 0.8011],
                ],
                [
                    [-0.5093, -0.7312, 0.9297],
                    [1.1871, -0.0757, -0.8522],
                    [0.0582, -1.0006, 0.9050],
                    [0.4631, -0.9681, -0.6372],
                    [0.1152, 0.5177, 0.3457],
                ],
                [
                    [0.2923, 0.7533, -0.4524],
                    [-1.6096, 0.6572, -0.7415],
                    [0.7128, -0.2328, 0.4080],
                    [-0.4106, 0.0252, 0.9320],
                    [0.6322, 0.9878, 1.0303],
                ],
                [
                    [-0.2311, 1.4206, -0.1557],
                    [-0.8234, -0.8210, 0.7172],
                    [0.2048, -0.2482, 1.7082],
                    [-0.2408, -0.4810, 1.4828],
                    [0.2252, 0.4402, 0.0496],
                ],
            ],
        ])
        .to_device(&mut ctx);

        let y_gt = Tensor::new([
            [
                [
                    [1.7142, 1.1953, 1.4118],
                    [1.7142, 1.8230, 1.4118],
                    [0.8096, 1.8230, 0.3624],
                ],
                [
                    [1.8615, 1.1953, 0.7561],
                    [1.8615, 1.8230, 0.4055],
                    [0.8096, 1.8230, 0.4055],
                ],
                [
                    [1.8615, 2.1217, 1.4202],
                    [1.8615, 2.1217, 0.4055],
                    [0.8096, 1.4073, 0.9963],
                ],
            ],
            [
                [
                    [1.8998, 0.9345, 1.3045],
                    [1.8998, 2.0919, 1.2738],
                    [0.9705, 2.0919, 0.9050],
                ],
                [
                    [1.1871, 0.7533, 1.3045],
                    [1.1871, 2.0919, 1.2738],
                    [0.9705, 2.0919, 1.0303],
                ],
                [
                    [1.1871, 1.4206, 1.7082],
                    [1.1871, 0.6572, 1.7082],
                    [0.7128, 0.9878, 1.7082],
                ],
            ],
        ]);

        let x = Fun::new(x);

        let y = max_pool_2d(&x, 3, 1, 0, 1);

        println!("{:?}", y.eval(&mut ctx));

        assert!(Tensor::all_close(&y.eval(&mut ctx), &y_gt, 0.01));
    }
}
