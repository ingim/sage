use crate::layers::{
    AdaptiveAvgPool2d, AvgPool2d, BatchNorm2d, Conv2d, Dense, Flatten, Layer, Parameter, Relu,
    Sequential,
};
use crate::ops::core::concat;
use crate::session::context::Context;
use crate::var::Function;
use itertools::Itertools;

#[derive(Copy, Clone)]
pub struct DenseNetConfig {
    chan_input: usize,
    depth: usize,
    growth_rate: usize,
    batch_norm_eps: f32,
    dropout_prob: f32,
    num_classes: usize,
}

impl DenseNetConfig {
    pub fn d121(chan_input: usize, num_classes: usize) -> Self {
        DenseNetConfig {
            chan_input,
            num_classes,
            depth: 121,
            growth_rate: 12,
            batch_norm_eps: 0.0001,
            dropout_prob: 0.2,
        }
    }

    pub fn d169(chan_input: usize, num_classes: usize) -> Self {
        DenseNetConfig {
            chan_input,
            num_classes,
            depth: 169,
            growth_rate: 24,
            batch_norm_eps: 0.0001,
            dropout_prob: 0.2,
        }
    }

    pub fn d201(chan_input: usize, num_classes: usize) -> Self {
        DenseNetConfig {
            chan_input,
            num_classes,
            depth: 201,
            growth_rate: 12,
            batch_norm_eps: 0.0001,
            dropout_prob: 0.2,
        }
    }
}

pub struct DenseNet(Sequential);

impl DenseNet {
    pub fn new(config: DenseNetConfig) -> Self {
        let mut in_planes = 2 * config.growth_rate;

        let mut model = Sequential::new();
        model.add(Conv2d::with(config.chan_input, in_planes, 3, 1, 1, 1));

        let n = (config.depth - 4) / 6;

        for i in 0..3 {
            model.add(Self::dense_layer(n, in_planes, config));
            in_planes += n * config.growth_rate;

            if i < 2 {
                model.add(Self::transition_layer(in_planes, in_planes / 2, config));
                in_planes /= 2;
            }
        }
        //model.add(BatchNorm2d::new(in_planes, config.batch_norm_eps));
        model.add(Relu);
        model.add(AdaptiveAvgPool2d::new([1, 1]));
        model.add(Flatten);
        model.add(Dense::new(in_planes, config.num_classes));

        DenseNet(model)
    }

    fn transition_layer(in_planes: usize, out_planes: usize, config: DenseNetConfig) -> Sequential {
        Sequential::new()
            //.with(BatchNorm2d::new(in_planes, config.batch_norm_eps))
            .with(Relu)
            .with(Conv2d::with(in_planes, out_planes, 1, 1, 0, 1))
            .with(AvgPool2d::new(2))
    }

    fn dense_layer(num_layers: usize, in_planes: usize, config: DenseNetConfig) -> Sequential {
        let mut layer = Sequential::new();
        for i in 0..num_layers {
            layer.add(BottleneckLayer::new(
                in_planes + i * config.growth_rate,
                config.growth_rate,
                config,
            ));
        }
        layer
    }
}

impl Parameter for DenseNet {
    fn init(&mut self, ctx: &mut Context, level: usize) {
        let indent = (0..level).map(|_| "----").join("");
        println!("{indent} resnet");
        self.0.init(ctx, level);
    }
    fn params<'a>(&'a self, p: &mut Vec<&'a Function>) {
        self.0.params(p);
    }
}

impl Layer for DenseNet {
    fn pass(&self, x: &Function) -> Function {
        self.0.pass(x)
    }
}

struct BottleneckLayer(Sequential);

impl BottleneckLayer {
    pub fn new(in_planes: usize, out_planes: usize, config: DenseNetConfig) -> Self {
        let inter_planes = out_planes * 4;

        let layer = Sequential::new()
            //.add(BatchNorm2d::new(in_planes, config.batch_norm_eps));
            .with(Relu)
            .with(Conv2d::with(in_planes, inter_planes, 1, 1, 0, 1))
            //.add(BatchNorm2d::new(inter_planes, config.batch_norm_eps))
            .with(Conv2d::with(inter_planes, out_planes, 3, 1, 1, 1));

        BottleneckLayer(layer)
    }
}

impl Parameter for BottleneckLayer {
    fn init(&mut self, ctx: &mut Context, level: usize) {
        let indent = (0..level).map(|_| "----").join("");
        println!("{indent} resnet");
        self.0.init(ctx, level);
    }
    fn params<'a>(&'a self, p: &mut Vec<&'a Function>) {
        self.0.params(p);
    }
}

impl Layer for BottleneckLayer {
    fn pass(&self, x: &Function) -> Function {
        let y = self.0.pass(x);
        //println!("{:?}, {:?}", x.extents(), y.extents());
        // key idea of the DenseNet
        concat([x, &y], 3)
    }
}
