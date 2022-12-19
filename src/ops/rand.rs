use crate::error::Error;
use crate::ops::core::div_up;
use crate::ops::{Category, Compose, Composer};
use crate::session::context::Context;
use crate::shape::{Array, Extent, Shape};
use crate::tensor::data::DataType;
use crate::tensor::{Tensor, TensorDesc};
use crate::var::Fun;
use rand::Rng;

#[derive(Clone, Debug)]
struct Uniform {
    output: TensorDesc,
}

#[derive(Clone, Debug)]
struct Normal {
    output: TensorDesc,
}

pub fn uniform<E>(extents: E) -> Fun
where
    E: Extent,
{
    Fun::from_nullary_op(Uniform {
        output: TensorDesc::new(extents, DataType::Float),
    })
}

pub fn normal<E>(extents: E) -> Fun
where
    E: Extent,
{
    Fun::from_nullary_op(Normal {
        output: TensorDesc::new(extents, DataType::Float),
    })
}

fn xorshift128() -> &'static str {
    r#"
    float xorshift128(uint seed_x, uint seed_y) {
    
        const uint t = seed_x ^ (seed_x << 11);
        const uint rndint = seed_y ^ (seed_y >> 19) ^ (t ^ (t >> 8));
       
        return rndint * 2.3283064e-10;
    }"#
}

impl Compose<0> for Uniform {
    fn input(&self) -> &[TensorDesc; 0] {
        &[]
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, _: [&Fun; 0], _: &Fun, _: &Fun) -> [Option<Fun>; 0] {
        []
    }

    fn compute(&self, _: [&Tensor; 0], ctx: &mut Context) -> Result<Tensor, Error> {
        let mut rng = rand::thread_rng();
        let y = Tensor::uninit2(self.output(), ctx)?;

        // Implements xorshift128 PRNG
        // https://en.wikipedia.org/wiki/Xorshift
        let p = ctx.get_program(format!(
            r#"
            {xorshift}
            __kernel void uniform(__global float *y,
                __private const uint seed_x,
                __private const uint seed_y) {{
                const uint gid = get_global_id(0);
                y[gid] = xorshift128(seed_x + gid, seed_y+ gid);
            }}"#,
            xorshift = xorshift128()
        ));

        p.kernel("uniform")
            .arg_tensor(&y)
            .arg(rng.gen::<u32>())
            .arg(rng.gen::<u32>())
            .global_work_size(y.size())
            .launch()
            .map_err(|e| Error::Device(e))?;

        Ok(y)
    }
}

impl Compose<0> for Normal {
    fn input(&self) -> &[TensorDesc; 0] {
        &[]
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, _: [&Fun; 0], _: &Fun, _: &Fun) -> [Option<Fun>; 0] {
        []
    }
    fn compute(&self, _: [&Tensor; 0], ctx: &mut Context) -> Result<Tensor, Error> {
        let mut rng = rand::thread_rng();
        let y = Tensor::uninit2(self.output(), ctx)?;

        // Box-Muller transform
        // https://en.wikipedia.org/wiki/Box–Muller_transform
        let p = ctx.get_program(format!(
            r#"
            {xorshift}
            __kernel void normal(__global float *y,
                __private const uint seed_x,
                __private const uint seed_y) {{

                const uint gid = get_global_id(0);

                const float u1 = xorshift128(seed_x + gid * 2, seed_y);
                const float u2 = xorshift128(seed_x + gid * 2 + 1, seed_y);

                const float rsq = -2 * log(u1);
                const float theta = 2 * M_PI * u2;
                const float z1 = sqrt(rsq) * cos(theta);
                const float z2 = sqrt(rsq) * sin(theta);

                y[gid * 2] = z1;
                y[gid * 2 + 1] = z2;
            }}"#,
            xorshift = xorshift128()
        ));

        p.kernel("normal")
            .arg_tensor(&y)
            .arg(rng.gen::<u32>())
            .arg(rng.gen::<u32>())
            .global_work_size(div_up(y.size(), 2))
            .launch()
            .map_err(|e| Error::Device(e))?;

        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::rand::uniform;
    use crate::session::context::Context;

    #[test]
    fn test_uniform() {
        let mut ctx = Context::new();

        let y = uniform([3, 3]).eval(&mut ctx);
    }
}
