use crate::ops::map::eq;
use crate::var::{Fun, ToFun};

pub fn accuracy<V1, V2>(logits: V1, labels: V2) -> Fun
where
    V1: ToFun,
    V2: ToFun,
{
    let logits = logits.to_fun().argmax(1, true);
    let labels = labels.to_fun();

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
