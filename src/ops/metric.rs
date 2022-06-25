use crate::ops::map::eq;
use crate::var::{Var, Variable};

pub fn accuracy<V1, V2>(logits: V1, labels: V2) -> Var
where
    V1: Variable,
    V2: Variable,
{
    let logits = logits.into_var().argmax(1, true);
    let labels = labels.into_var();

    // println!("{:?}",logits );
    // println!("{:?}",labels );

    eq(&logits, labels).sum(0, false).float() / (logits.size() as f32)
}
//
//
// pub fn accuracy<V1, V2>(logits: V1, labels: V2) -> Var
// where
//     V1: Variable,
//     V2: Variable,
// {
//     let logits = logits.into_var().argmax(1, false);
//     let labels = labels.into_var().argmax(1, false);
//
//     eq(&logits, labels).sum(0, false).float() / (logits.size() as f32)
// }
