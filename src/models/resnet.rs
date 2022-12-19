use crate::layers::{
    AdaptiveAvgPool2d, AvgPool2d, BatchNorm2d, Conv2d, Dense, Flatten, Layer, MaxPool2d, Parameter,
    Relu, Sequential,
};
use crate::ops::nn::relu;
use crate::session::context::Context;
use crate::var::Fun;
use itertools::Itertools;
use std::time::Instant;

#[derive(Clone, Copy)]
pub struct ResNetConfig {
    chan_input: usize,
    num_classes: usize,
    num_blocks: [usize; 4],
    expansion: usize,
    use_bottleneck: bool,
}

impl ResNetConfig {
    pub fn d18(chan_input: usize, num_classes: usize) -> Self {
        ResNetConfig {
            chan_input,
            num_classes,
            num_blocks: [2, 2, 2, 2],
            expansion: 1,
            use_bottleneck: false,
        }
    }

    pub fn d34(chan_input: usize, num_classes: usize) -> Self {
        ResNetConfig {
            chan_input,
            num_classes,
            num_blocks: [3, 4, 6, 3],
            expansion: 1,
            use_bottleneck: false,
        }
    }

    pub fn d50(chan_input: usize, num_classes: usize) -> Self {
        ResNetConfig {
            chan_input,
            num_classes,
            num_blocks: [3, 4, 6, 3],
            expansion: 4,
            use_bottleneck: true,
        }
    }

    pub fn d101(chan_input: usize, num_classes: usize) -> Self {
        ResNetConfig {
            chan_input,
            num_classes,
            num_blocks: [3, 4, 23, 3],
            expansion: 4,
            use_bottleneck: true,
        }
    }

    pub fn d152(chan_input: usize, num_classes: usize) -> Self {
        ResNetConfig {
            chan_input,
            num_classes,
            num_blocks: [3, 8, 36, 3],
            expansion: 4,
            use_bottleneck: true,
        }
    }
}

pub struct ResNet(Sequential);

impl ResNet {
    pub fn new(config: ResNetConfig) -> Self {
        let mut model = Sequential::new();

        let mut in_planes = 64;
        let mut out_planes = in_planes;

        model.add(Conv2d::with(config.chan_input, in_planes, 7, 2, 3, 1));
        //pass.add(BatchNorm2d::new(in_planes, config.eps));
        model.add(Relu);
        model.add(MaxPool2d::with(3, 2, 1, 1));

        for i in 0..4 {
            let stride = if i == 0 { 1 } else { 2 };
            model.add(Self::residual_layers(
                config.num_blocks[i],
                in_planes,
                out_planes,
                stride,
                config,
            ));
            in_planes = out_planes * config.expansion;
            out_planes *= 2;
        }

        model.add(AdaptiveAvgPool2d::new([1, 1]));
        model.add(Flatten);
        model.add(Dense::new(512 * config.expansion, config.num_classes));

        ResNet(model)
    }

    fn residual_layers(
        num_blocks: usize,
        in_planes: usize,
        out_planes: usize,
        stride: usize,
        config: ResNetConfig,
    ) -> Sequential {
        let mut layer = Sequential::new();

        let mut in_planes = in_planes;

        for i in 0..num_blocks {
            let stride = if i == 0 { stride } else { 1 };

            if config.use_bottleneck {
                layer.add(BottleneckLayer::new(in_planes, out_planes, stride, config));
            } else {
                layer.add(BasicLayer::new(in_planes, out_planes, stride, config));
            }

            in_planes = out_planes * config.expansion;
        }

        layer
    }
}

impl Parameter for ResNet {
    fn init(&mut self, ctx: &mut Context, level: usize) {
        let indent = (0..level).map(|_| "----").join("");
        println!("{indent} resnet");
        self.0.init(ctx, level);
    }
    fn params<'a>(&'a self, p: &mut Vec<&'a Fun>) {
        self.0.params(p);
    }
}

impl Layer for ResNet {
    fn pass(&self, x: &Fun) -> Fun {
        self.0.pass(x)
    }
}

struct BasicLayer {
    pass: Sequential,
    downsample: Option<Sequential>,
}

impl BasicLayer {
    pub fn new(in_planes: usize, planes: usize, stride: usize, config: ResNetConfig) -> Self {
        let out_planes = config.expansion * planes;

        let mut layer = Sequential::new()
            .with(Conv2d::with(in_planes, planes, 3, stride, 1, 1))
            //.with(BatchNorm2d::new(planes, config.eps))
            .with(Relu)
            .with(Conv2d::with(planes, out_planes, 3, 1, 1, 1));
        //.with(BatchNorm2d::new(out_planes, config.eps))

        let downsample = if stride != 1 || in_planes != out_planes {
            Some(Sequential::new().with(Conv2d::with(in_planes, out_planes, 1, stride, 0, 1)))
        } else {
            None
        };

        BasicLayer {
            pass: layer,
            downsample,
        }
    }
}

struct BottleneckLayer {
    pass: Sequential,
    downsample: Option<Sequential>,
}

impl BottleneckLayer {
    fn new(in_planes: usize, planes: usize, stride: usize, config: ResNetConfig) -> Self {
        let out_planes = config.expansion * planes;

        let layer = Sequential::new()
            .with(Conv2d::new(in_planes, planes, 1))
            //.with(BatchNorm2d::new(planes, config.eps));
            .with(Relu)
            .with(Conv2d::with(planes, planes, 3, stride, 1, 1))
            //.with(BatchNorm2d::new(planes, config.eps));
            .with(Relu)
            .with(Conv2d::new(planes, out_planes, 1));
        //.with(BatchNorm2d::new(out_planes, config.eps));

        let downsample = if stride != 1 || in_planes != out_planes {
            Some(Sequential::new().with(Conv2d::with(in_planes, out_planes, 1, stride, 0, 1)))
            //.with(BatchNorm2d::new(out_planes, config.eps));
        } else {
            None
        };

        BottleneckLayer {
            pass: layer,
            downsample,
        }
    }
}

impl Parameter for BottleneckLayer {
    fn init(&mut self, ctx: &mut Context, level: usize) {
        let indent = (0..level).map(|_| "----").join("");
        let now = Instant::now();
        self.pass.init(ctx, level);

        println!("{indent} basic_layer.pass ({:.2?})", now.elapsed());

        if let Some(d) = &mut self.downsample {
            let now = Instant::now();
            d.init(ctx, level);
            println!("{indent} basic_layer.downsample ({:.2?})", now.elapsed());
        }
    }

    fn params<'a>(&'a self, p: &mut Vec<&'a Fun>) {
        self.pass.params(p);
        if let Some(d) = &self.downsample {
            d.params(p);
        }
    }
}

impl Layer for BottleneckLayer {
    fn pass(&self, x: &Fun) -> Fun {
        let y_long = self.pass.pass(x);
        let y = if let Some(d) = &self.downsample {
            y_long + d.pass(x)
        } else {
            y_long + x
        };
        relu(y)
    }
}

impl Parameter for BasicLayer {
    fn init(&mut self, ctx: &mut Context, level: usize) {
        let indent = (0..level).map(|_| "----").join("");
        let now = Instant::now();
        self.pass.init(ctx, level);

        println!("{indent} basic_layer.pass ({:.2?})", now.elapsed());

        if let Some(d) = &mut self.downsample {
            let now = Instant::now();
            d.init(ctx, level);
            println!("{indent} basic_layer.downsample ({:.2?})", now.elapsed());
        }
    }

    fn params<'a>(&'a self, p: &mut Vec<&'a Fun>) {
        self.pass.params(p);
        if let Some(d) = &self.downsample {
            d.params(p);
        }
    }
}

impl Layer for BasicLayer {
    fn pass(&self, x: &Fun) -> Fun {
        let y_long = self.pass.pass(x);
        let y = if let Some(d) = &self.downsample {
            y_long + d.pass(x)
        } else {
            y_long + x
        };
        relu(y)
    }
}
