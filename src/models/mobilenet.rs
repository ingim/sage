use crate::layers::{
    AdaptiveAvgPool2d, AvgPool2d, BatchNorm2d, Conv2d, Dense, Flatten, Layer, Parameter, Relu,
    Sequential,
};
use crate::ops::core::concat;
use crate::session::context::Context;
use crate::var::Function;
use std::cmp::max;

fn make_divisible(val: usize, div: usize, min_val: usize) -> usize {
    let new_val = max(min_val, (val + div / 2) / div * div);

    if (new_val as f32) < (0.9 * (val as f32)) {
        new_val + div
    } else {
        new_val
    }
}

#[derive(Clone, Copy)]
pub struct MobileNetV2Config {
    width_mul: usize,
    chan_in: usize,
    num_classes: usize,
}

impl MobileNetV2Config {
    pub fn new(chan_in: usize, num_classes: usize) -> Self {
        MobileNetV2Config {
            width_mul: 1,
            chan_in,
            num_classes,
        }
    }
}

pub struct MobileNetV2(Sequential);

impl MobileNetV2 {
    pub fn new(config: MobileNetV2Config) -> Self {
        let mut chan_in = make_divisible(32 * config.width_mul, 8, 8);

        // t, c, n, s
        let cfg = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ];

        let mut model = Sequential::new();
        model.add(Conv2d::with(config.chan_in, chan_in, 3, 2, 1, 1));
        //model.add(BatchNorm2d::new(chan_in, 0.001));
        model.add(Relu);

        for (t, c, n, s) in cfg {
            let chan_out = make_divisible(c * config.width_mul, 8, 8);

            for i in 0..n {
                let stride = if i == 0 { s } else { 1 };
                model.add(InvertedResidual::new(chan_in, chan_out, stride, t));
                chan_in = chan_out;
            }
        }

        let chan_out = make_divisible(1280 * config.width_mul, 8, 8);

        model.add(Conv2d::new(chan_in, chan_out, 1));
        //model.add(BatchNorm2d::new(chan_out, 0.001));
        model.add(Relu);
        model.add(AdaptiveAvgPool2d::new([1, 1]));
        model.add(Flatten);
        model.add(Dense::new(chan_out, config.num_classes));

        MobileNetV2(model)
    }
}

impl Parameter for MobileNetV2 {
    fn init(&mut self, ctx: &mut Context, level: usize) {
        self.0.init(ctx, level)
    }
    fn params<'a>(&'a self, p: &mut Vec<&'a Function>) {
        self.0.params(p);
    }
}

impl Layer for MobileNetV2 {
    fn pass(&self, x: &Function) -> Function {
        self.0.pass(x)
    }
}

struct InvertedResidual(Sequential, bool);

impl InvertedResidual {
    pub fn new(inp: usize, oup: usize, stride: usize, expand_ratio: usize) -> Self {
        let hidden_dim = inp * expand_ratio;

        let mut layer = Sequential::new();

        if expand_ratio != 1 {
            layer.add(Conv2d::with(inp, hidden_dim, 1, 1, 0, 1));
            //layer.add(BatchNorm2d::new(hidden_dim, 0.001));
            layer.add(Relu);
        }

        layer.add(Conv2d::with(hidden_dim, hidden_dim, 3, stride, 0, 0));
        //layer.add(BatchNorm2d::new(hidden_dim, 0.001));
        layer.add(Relu);
        layer.add(Conv2d::with(hidden_dim, oup, 1, 1, 0, 1));
        //layer.add(BatchNorm2d::new(oup, 0.001));

        InvertedResidual(layer, stride == 1 && inp == oup)
    }
}

impl Parameter for InvertedResidual {
    fn init(&mut self, ctx: &mut Context, level: usize) {
        self.0.init(ctx, level)
    }
    fn params<'a>(&'a self, p: &mut Vec<&'a Function>) {
        self.0.params(p);
    }
}

impl Layer for InvertedResidual {
    fn pass(&self, x: &Function) -> Function {
        if self.1 {
            x + self.0.pass(x)
        } else {
            self.0.pass(x)
        }
    }
}
