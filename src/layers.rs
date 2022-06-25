use crate::ops::conv::{avg_pool_2d, batch_norm_2d, conv_2d, max_pool_2d};
use crate::ops::core::div_up;
use crate::ops::nn::{layer_norm, relu, softmax};
use crate::session::context::Context;
use crate::shape::{Array, Axes, Extent, SizedExtent};
use crate::tensor::data::DataType;
use crate::tensor::init::{kaiming_normal, kaiming_uniform};
use crate::tensor::Tensor;
use crate::var::Var;
use itertools::Itertools;
use std::fmt::{Debug, Formatter};
use std::time::Instant;

pub trait Parameter {
    fn init(&mut self, ctx: &mut Context, level: usize) {}
    fn params<'a>(&'a self, p: &mut Vec<&'a Var>) {}
}

pub trait Layer: Parameter {
    fn pass(&self, x: &Var) -> Var;
}

pub struct Filter<const N: usize> {
    pub kernel_size: [usize; N],
    pub stride: [usize; N],
    pub padding: [usize; N],
    pub dilation: [usize; N],
}

impl<const N: usize> Filter<N> {
    pub fn new<E>(kernel_size: E) -> Self
    where
        E: SizedExtent<N>,
    {
        Self {
            kernel_size: kernel_size.to_arr(),
            stride: [1; N],
            padding: [0; N],
            dilation: [1; N],
        }
    }

    pub fn with<E>(kernel_size: E, stride: E, padding: E, dilation: E) -> Self
    where
        E: SizedExtent<N>,
    {
        Self {
            kernel_size: kernel_size.to_arr(),
            stride: stride.to_arr(),
            padding: padding.to_arr(),
            dilation: dilation.to_arr(),
        }
    }
}

pub struct MaxPool2d(Filter<2>);

impl MaxPool2d {
    pub fn new<E>(kernel_size: E) -> Self
    where
        E: SizedExtent<2>,
    {
        let kernel_size = kernel_size.to_arr();
        Self(Filter::with(kernel_size, kernel_size, [0, 0], [1, 1]))
    }

    pub fn with<E>(kernel_size: E, stride: E, padding: E, dilation: E) -> Self
    where
        E: SizedExtent<2>,
    {
        Self(Filter::with(kernel_size, stride, padding, dilation))
    }
}

impl Parameter for MaxPool2d {}

impl Layer for MaxPool2d {
    fn pass(&self, x: &Var) -> Var {
        max_pool_2d(
            x,
            self.0.kernel_size,
            self.0.stride,
            self.0.padding,
            self.0.dilation,
        )
        .name("max_pool_2d")
    }
}

pub struct AvgPool2d(Filter<2>);

impl AvgPool2d {
    pub fn new<E>(kernel_size: E) -> Self
    where
        E: SizedExtent<2>,
    {
        let kernel_size = kernel_size.to_arr();
        Self(Filter::with(kernel_size, kernel_size, [0, 0], [1, 1]))
    }

    pub fn with<E>(kernel_size: E, stride: E, padding: E, dilation: E) -> Self
    where
        E: SizedExtent<2>,
    {
        Self(Filter::with(kernel_size, stride, padding, dilation))
    }
}

impl Parameter for AvgPool2d {}

impl Layer for AvgPool2d {
    fn pass(&self, x: &Var) -> Var {
        avg_pool_2d(
            x,
            self.0.kernel_size,
            self.0.stride,
            self.0.padding,
            self.0.dilation,
        )
        .name("avg_pool_2d")
    }
}

pub struct AdaptiveAvgPool2d {
    extents: [usize; 2],
}

impl AdaptiveAvgPool2d {
    pub fn new<E>(extents: E) -> Self
    where
        E: SizedExtent<2>,
    {
        AdaptiveAvgPool2d {
            extents: extents.to_arr(),
        }
    }
}

impl Parameter for AdaptiveAvgPool2d {}

impl Layer for AdaptiveAvgPool2d {
    fn pass(&self, x: &Var) -> Var {
        let (inp_w, inp_h) = (x.extent(2), x.extent(1));
        let (out_w, out_h) = (self.extents[0], self.extents[1]);

        let stride = [inp_w / out_w, inp_h / out_h];
        let ker = [
            inp_w - (out_w - 1) * stride[0],
            inp_h - (out_h - 1) * stride[1],
        ];

        avg_pool_2d(x, ker, stride, [0, 0], [1, 1]).name("adaptive_avg_pool_2d")
    }
}

pub struct AdaptiveMaxPool2d {
    extents: [usize; 2],
}

impl AdaptiveMaxPool2d {
    pub fn new<E>(extents: E) -> Self
    where
        E: SizedExtent<2>,
    {
        AdaptiveMaxPool2d {
            extents: extents.to_arr(),
        }
    }
}

impl Parameter for AdaptiveMaxPool2d {}

impl Layer for AdaptiveMaxPool2d {
    fn pass(&self, x: &Var) -> Var {
        let (inp_w, inp_h) = (x.extent(2), x.extent(1));
        let (out_w, out_h) = (self.extents[0], self.extents[1]);

        let stride = [inp_w / out_w, inp_h / out_h];
        let ker = [
            inp_w - (out_w - 1) * stride[0],
            inp_h - (out_h - 1) * stride[1],
        ];

        max_pool_2d(x, ker, stride, [0, 0], [1, 1]).name("adaptive_max_pool_2d")
    }
}

pub struct Relu;

impl Parameter for Relu {}

impl Layer for Relu {
    fn pass(&self, x: &Var) -> Var {
        relu(x)
    }
}

pub struct Flatten;

impl Parameter for Flatten {}

impl Layer for Flatten {
    fn pass(&self, x: &Var) -> Var {
        x.view([x.extent(0) as i32, -1]).name("flatten")
    }
}

pub struct Dense {
    pub weight: Var,
    pub bias: Var,
}

impl Dense {
    pub fn new(dim_in: usize, dim_out: usize) -> Self {
        Dense {
            weight: Var::empty([dim_in, dim_out], DataType::Float).name("dense.weight"),
            bias: Var::empty(dim_out, DataType::Float).name("dense.bias"),
        }
    }
}

impl Parameter for Dense {
    fn init(&mut self, ctx: &mut Context, level: usize) {
        let indent = (0..level).map(|_| "----").join("");
        let now = Instant::now();

        self.weight
            .set(kaiming_uniform(self.weight.extents(), 1.0).to_device(ctx));
        println!("{indent} {:?} ({:.2?})", &self.weight, now.elapsed());
        let now = Instant::now();

        self.bias
            .set(Tensor::zeros(self.bias.extents()).to_device(ctx));
        println!("{indent} {:?} ({:.2?})", &self.bias, now.elapsed());
    }

    fn params<'a>(&'a self, p: &mut Vec<&'a Var>) {
        p.push(&self.weight);
        p.push(&self.bias);
    }
}

impl Layer for Dense {
    fn pass(&self, x: &Var) -> Var {
        (x.matmul(&self.weight) + &self.bias).name("dense")
    }
}

pub struct Softmax {
    pub axes: Array,
}

impl Softmax {
    pub fn new<A>(axes: A) -> Self
    where
        A: Axes,
    {
        Softmax {
            axes: axes.to_arr(99).unwrap(),
        }
    }
}

impl Parameter for Softmax {}

impl Layer for Softmax {
    fn pass(&self, x: &Var) -> Var {
        //println!("{:?}", x.extents());
        //println!("{:?}", &self.axes);

        softmax(x, &self.axes).name("softmax")
    }
}

#[derive(Default)]
pub struct Sequential {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new() -> Self {
        Sequential::default()
    }

    pub fn with<L>(mut self, layer: L) -> Self
    where
        L: Layer + 'static,
    {
        self.add(layer);
        self
    }

    pub fn add<L>(&mut self, layer: L) -> &mut Self
    where
        L: Layer + 'static,
    {
        self.layers.push(Box::new(layer));
        self
    }
}

impl Parameter for Sequential {
    fn init(&mut self, ctx: &mut Context, level: usize) {
        let indent = (0..level).map(|_| "----").join("");
        println!("{indent} sequence of {}", self.layers.len());
        for layer in self.layers.iter_mut() {
            layer.init(ctx, level + 1);
        }
    }
    fn params<'a>(&'a self, p: &mut Vec<&'a Var>) {
        for layer in self.layers.iter() {
            layer.params(p)
        }
    }
}

impl Layer for Sequential {
    fn pass(&self, x: &Var) -> Var {
        let mut x = x.clone();
        for layer in self.layers.iter() {
            x = layer.pass(&x);
        }
        x.name("sequential")
    }
}

pub struct Conv2d {
    pub weight: Var,
    pub bias: Var,

    pub filter: Filter<2>,
}

impl Conv2d {
    pub fn new<E>(chan_in: usize, chan_out: usize, kernel_size: E) -> Self
    where
        E: SizedExtent<2>,
    {
        Self::with(
            chan_in,
            chan_out,
            [kernel_size.at(0), kernel_size.at(1)],
            [1, 1],
            [0, 0],
            [1, 1],
        )
    }

    pub fn with<E>(
        chan_in: usize,
        chan_out: usize,
        kernel_size: E,
        stride: E,
        padding: E,
        dilation: E,
    ) -> Self
    where
        E: SizedExtent<2>,
    {
        Conv2d {
            //(KH, KW, C, OC)
            weight: Var::empty(
                [kernel_size.at(1), kernel_size.at(0), chan_in, chan_out],
                DataType::Float,
            )
            .name("conv2d.weight"),
            bias: Var::empty(chan_out, DataType::Float).name("conv2d.bias"),
            filter: Filter::with(kernel_size, stride, padding, dilation),
        }
    }
}

impl Parameter for Conv2d {
    fn init(&mut self, ctx: &mut Context, level: usize) {
        let indent = (0..level).map(|_| "----").join("");
        let now = Instant::now();
        self.weight
            .set(kaiming_normal(self.weight.extents(), 1.0).to_device(ctx));
        println!("{indent} {:?} ({:.2?})", &self.weight, now.elapsed());
        let now = Instant::now();

        self.bias
            .set(Tensor::zeros(self.bias.extents()).to_device(ctx));
        println!("{indent} {:?} ({:.2?})", &self.bias, now.elapsed());
    }

    fn params<'a>(&'a self, p: &mut Vec<&'a Var>) {
        p.push(&self.weight);
        p.push(&self.bias);
    }
}

impl Layer for Conv2d {
    fn pass(&self, x: &Var) -> Var {
        (conv_2d(
            x,
            &self.weight,
            self.filter.stride,
            self.filter.padding,
            self.filter.dilation,
        ) + &self.bias)
            .name("conv_2d")
    }
}

pub struct Embedding {
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub weight: Var,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        Embedding {
            num_embeddings,
            embedding_dim,
            weight: Var::empty([num_embeddings, embedding_dim], DataType::Float)
                .name("embedding.weight"),
        }
    }
}

impl Layer for Embedding {
    fn pass(&self, idx: &Var) -> Var {
        // (512, 768) g (64, 768) -> (64, 768)
        //println!("{:?}", idx.extents());
        //println!("{:?}", [idx.extent(0), self.embedding_dim]);

        // batched
        if idx.rank() > 1 {
            let batch = idx.extent(0);
            let seq_len = idx.extent(1);

            // (300, 768) g (64, 20, 1)  -> (64, 768)
            // (300, 768) g (64, 20, 768)

            // (64, 20) -> (64*20) -> (64*20, 1) -> (64*20, 768)
            let idx = idx
                .view([batch * seq_len, 1])
                .expand([batch * seq_len, self.embedding_dim]);

            // (300, 768) g (64*20, 768) -> (64*20, 768)
            self.weight
                .gather(idx, 0)
                .view([batch, seq_len, self.embedding_dim])
                .name("embedding")
        } else {
            let seq_len = idx.extent(0);

            // (300, 768) g (20, 1)  -> (20, 768)
            // (300, 768) g (20, 768)

            // (64, 20) -> (64*20) -> (64*20, 1) -> (64*20, 768)
            let idx = idx.view([seq_len, 1]).expand([seq_len, self.embedding_dim]);

            // (300, 768) g (64*20, 768) -> (64*20, 768)
            self.weight
                .gather(idx, 0)
                .view([seq_len, self.embedding_dim])
                .name("embedding")
        }
    }
}

impl Parameter for Embedding {
    fn init(&mut self, ctx: &mut Context, level: usize) {
        let indent = (0..level).map(|_| "----").join("");
        let now = Instant::now();
        self.weight
            .set(Tensor::randn(self.weight.extents()).to_device(ctx));
        println!("{indent} {:?} ({:.2?})", &self.weight, now.elapsed());
    }

    fn params<'a>(&'a self, p: &mut Vec<&'a Var>) {
        p.push(&self.weight);
    }
}

pub struct BatchNorm2d {
    num_features: usize,
    eps: f32,

    gamma: Var,
    beta: Var,

    // average batch mean
    running_mean: Var,

    // average batch variance
    running_var: Var,
}

impl BatchNorm2d {
    pub fn new(num_features: usize, eps: f32) -> Self {
        BatchNorm2d {
            num_features,
            eps,
            gamma: Var::empty([1], DataType::Float).name("batch_norm_2d.gamma"),
            beta: Var::empty([1], DataType::Float).name("batch_norm_2d.beta"),
            running_mean: Var::empty([num_features], DataType::Float)
                .name("batch_norm_2d.running_mean"),
            running_var: Var::empty([num_features], DataType::Float)
                .name("batch_norm_2d.running_var"),
        }
    }
}

impl Parameter for BatchNorm2d {
    fn init(&mut self, ctx: &mut Context, level: usize) {
        let indent = (0..level).map(|_| "----").join("");
        let now = Instant::now();
        self.gamma.set(Tensor::from_scalar(1, 1.0).to_device(ctx));
        println!("{indent} {:?} ({:.2?})", &self.gamma, now.elapsed());
        let now = Instant::now();
        self.beta.set(Tensor::from_scalar(1, 1.0).to_device(ctx));
        println!("{indent} {:?} ({:.2?})", &self.beta, now.elapsed());
        let now = Instant::now();
        self.running_mean
            .set(Tensor::ones(self.num_features).to_device(ctx));
        println!("{indent} {:?} ({:.2?})", &self.running_mean, now.elapsed());
        let now = Instant::now();
        self.running_var
            .set(Tensor::ones(self.num_features).to_device(ctx));
        println!("{indent} {:?} ({:.2?})", &self.running_var, now.elapsed());
        let now = Instant::now();
    }

    fn params<'a>(&'a self, p: &mut Vec<&'a Var>) {
        p.push(&self.gamma);
        p.push(&self.beta);
    }
}

impl Layer for BatchNorm2d {
    fn pass(&self, x: &Var) -> Var {
        batch_norm_2d(
            x,
            &self.gamma,
            &self.beta,
            &self.running_mean,
            &self.running_var,
            1e-05,
        )
        .name("batch_norm_2d")
    }
}

pub struct LayerNorm {
    axes: Array,
    eps: f32,
    gamma: Var,
    beta: Var,
}

impl LayerNorm {
    pub fn new<A>(axes: A, eps: f32) -> Self
    where
        A: Axes,
    {
        LayerNorm {
            axes: axes.to_arr(90).unwrap(),
            eps,
            gamma: Var::empty([1], DataType::Float).name("layer_norm.gamma"),
            beta: Var::empty([1], DataType::Float).name("layer_norm.beta"),
        }
    }
}

impl Parameter for LayerNorm {
    fn init(&mut self, ctx: &mut Context, level: usize) {
        let indent = (0..level).map(|_| "----").join("");
        let now = Instant::now();
        self.gamma.set(Tensor::from_scalar([1], 1.0).to_device(ctx));
        println!("{indent} {:?} ({:.2?})", &self.gamma, now.elapsed());
        let now = Instant::now();
        self.beta.set(Tensor::from_scalar([1], 1.0).to_device(ctx));
        println!("{indent} {:?} ({:.2?})", &self.beta, now.elapsed());
    }

    fn params<'a>(&'a self, p: &mut Vec<&'a Var>) {
        p.push(&self.gamma);
        p.push(&self.beta);
    }
}

impl Layer for LayerNorm {
    fn pass(&self, x: &Var) -> Var {
        layer_norm(x, &self.axes, &self.gamma, &self.beta, 1e-05).name("layer_norm")
    }
}

// dropout is currently not implemented
pub struct Dropout {
    pub prob: f32,
}

impl Dropout {
    pub fn new(prob: f32) -> Self {
        Dropout { prob }
    }
}

impl Parameter for Dropout {}

impl Layer for Dropout {
    fn pass(&self, x: &Var) -> Var {
        x.clone()
    }
}

#[cfg(test)]
mod tests {
    use crate::layers::{Embedding, Layer, Parameter};
    use crate::session::context::Context;
    use crate::tensor::Tensor;
    use crate::var::Variable;

    #[test]
    fn test_embedding() {
        let mut ctx = Context::new();

        let mut emb = Embedding::new(10, 24);
        emb.init(&mut ctx, 0);

        let idx = Tensor::new([0, 1, 2, 3]).to_device(&mut ctx).into_var();
        let y = emb.pass(&idx.view([4, 1]));
        let y_gt = emb.weight.slice(0, 0, 4);

        assert!(Tensor::all_close(
            &y.eval(&mut ctx),
            &y_gt.eval(&mut ctx),
            0.01
        ));
    }
}
