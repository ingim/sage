use crate::shape::Extent;
use rand_distr::{Normal, Uniform};

use crate::tensor::Tensor;

pub fn kaiming_uniform<E>(extents: E, gain: f32) -> Tensor
where
    E: Extent,
{
    let (fan_in, _) = fan_in_and_out(&extents);
    let std = gain * (1.0 / fan_in as f32).sqrt();
    let a = 3.0_f32.sqrt() * std;

    Tensor::from_dist(extents, Uniform::new(-a, a))
}

pub fn kaiming_normal<E>(extents: E, gain: f32) -> Tensor
where
    E: Extent,
{
    let (fan_in, _) = fan_in_and_out(&extents);
    let std = gain * (1.0 / fan_in as f32).sqrt();

    Tensor::from_dist(extents, Normal::new(0.0, std).unwrap())
}

fn fan_in_and_out<E>(extents: &E) -> (usize, usize)
where
    E: Extent,
{
    let extents = extents.to_arr(0).unwrap();

    if extents.len() < 2 {
        panic!("cannot compute.. shape too small");
    }

    let num_in_fmaps = extents[1];
    let num_out_fmaps = extents[0];

    let mut receptive_field_size = 1;

    if extents.len() > 2 {
        receptive_field_size = extents[2..].iter().fold(1, |a, b| a * (*b));
    }

    let fan_in = num_in_fmaps * receptive_field_size;
    let fan_out = num_out_fmaps * receptive_field_size;

    (fan_in, fan_out)
}
