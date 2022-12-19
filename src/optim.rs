use crate::ops::map::scalar;
use crate::session::context::Context;
use crate::tensor::Tensor;
use crate::var::Fun;
use std::collections::HashMap;
use std::time::Instant;

pub trait Optimizer {
    fn init(&mut self, ctx: &mut Context);

    fn update(&mut self, grads: &HashMap<&Fun, Fun>, ctx: &mut Context);
}

pub struct Sgd {
    lr: f32,
    momentum: f32,
    dampening: f32,
    weight_decay: f32,

    // Nestreov momentum: On the importance of initialization and momentum in deep learning
    nesterov: bool,

    b: HashMap<Fun, Tensor>, // hashmap to store momentum tensors
    g: HashMap<Fun, Tensor>,
}

// Stochastic Gradient Descent
impl Sgd {
    pub fn new(lr: f32) -> Sgd {
        Sgd {
            lr,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            b: HashMap::new(),
            g: HashMap::new(),
        }
    }
}

impl Optimizer for Sgd {
    fn init(&mut self, ctx: &mut Context) {
        self.b.clear();
        self.g.clear();
    }

    fn update(&mut self, grads: &HashMap<&Fun, Fun>, ctx: &mut Context) {
        //println!("****************************************************");

        for (&param, grad) in grads {
            if !param.is_data() {
                panic!("param is not data tensor");
            }

            if !ctx.data.contains_key(grad) {
                println!("warning: it is inefficient to evaluate gradients in the inside of an optimizer");
                grad.eval(ctx);
            }

            let mut g = if self.weight_decay != 0.0 {
                grad + param * self.weight_decay
            } else {
                grad.clone()
            };

            if self.momentum != 0.0 {
                let b = if !self.b.contains_key(grad) {
                    self.b.get(grad).unwrap() * self.momentum + g * (1.0 - self.dampening)
                } else {
                    g
                };
                self.b.insert(grad.clone(), b.eval(ctx));

                if self.nesterov && self.g.contains_key(grad) {
                    g = self.g.get(grad).unwrap() + b * self.momentum;
                } else {
                    g = b;
                }
                self.g.insert(grad.clone(), g.eval(ctx));
            }

            let param_new = param - &g * self.lr;

            // println!("{:?}", param.extents());
            // //println!("{:?}", param_new.shape());
            // println!("{:?}", grad_m.eval(ctx));
            // //println!("{:?}", param_new.eval(ctx));
            // println!("----------------------------");

            param.set(param_new.eval(ctx));
        }
    }
}

//  Adam: A Method for Stochastic Optimization
pub struct Adam {
    t: Tensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,

    m_t: HashMap<Fun, Tensor>, // hashmap to store momentum tensors
    v_t: HashMap<Fun, Tensor>,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Adam {
            t: Tensor::from_scalar(1, 1.0),
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-08,
            weight_decay: 0.0,
            m_t: HashMap::new(),
            v_t: HashMap::new(),
        }
    }
}

// eager-execution like mode
impl Optimizer for Adam {
    fn init(&mut self, ctx: &mut Context) {
        self.m_t.clear();
        self.v_t.clear();
        self.t = Tensor::from_scalar(1, 1.0).to_device(ctx);
    }

    fn update(&mut self, grads: &HashMap<&Fun, Fun>, ctx: &mut Context) {
        // let v1 = self.m_t.values().map(|t| t.size()).sum::<usize>() * 4 / (1024 * 1024);
        // let v2 = self.v_t.values().map(|t| t.size()).sum::<usize>() * 4 / (1024 * 1024);
        // println!("{} MB, {}, {}", v1 + v2, self.m_t.len(), self.v_t.len());

        for (&param, grad) in grads {
            if !param.is_data() {
                panic!("param is not data tensor");
            }

            if !ctx.data.contains_key(grad) {
                println!("warning: it is inefficient to evaluate gradients in the inside of an optimizer");
                grad.eval(ctx);
            }

            let g = if self.weight_decay != 0.0 {
                grad + param * self.weight_decay
            } else {
                grad.clone()
            };

            let m_t = if self.m_t.contains_key(grad) {
                self.m_t.get(grad).unwrap() * self.beta1 + &g * (1.0 - self.beta1)
            } else {
                scalar(0.0, grad.extents())
            };

            self.m_t.insert(grad.clone(), m_t.eval(ctx));

            let v_t = if self.v_t.contains_key(grad) {
                self.v_t.get(grad).unwrap() * self.beta2 + g.pow(2.0) * (1.0 - self.beta2)
            } else {
                scalar(0.0, grad.extents())
            };
            self.v_t.insert(grad.clone(), v_t.eval(ctx));

            let m_t_hat = m_t / (-scalar(self.beta1, 1).pow(&self.t) + 1.0);
            let v_t_hat = v_t / (-scalar(self.beta2, 1).pow(&self.t) + 1.0);

            let param_new = param - m_t_hat * self.lr / (v_t_hat.sqrt() + self.eps);

            param.set(param_new.eval(ctx));
        }

        self.t = (&self.t + 1.0).eval(ctx);
    }
}
