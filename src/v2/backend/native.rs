use std::collections::HashMap;
use crate::v2::backend::{Backend, TensorPrimitive};
use crate::v2::ir;
use crate::v2::shape::Shape;

pub struct Native {}


impl Backend for Native {
    type Tensor = Tensor;

    fn eval(f: ir::Graph, inputs: HashMap<ir::Node, Self::Tensor>, outputs: Vec<ir::Node>) -> Vec<Self::Tensor> {

        // first optimize the graph
        let mut stack = outputs.clone();

        let mut visited = HashMap::<ir::Node, Self::Tensor>::new();
        let mut arg_buf = Vec::<Self::Tensor>::with_capacity(3);

        while let Some(node) = stack.pop() {
            let args = f.edges_in(node);

            let mut unvisited = args.iter().filter(|arg| !visited.contains_key(*arg)).peekable();

            if unvisited.peek().is_some() {
                stack.push(node);
                stack.extend(unvisited);
            } else {
                args.iter().map(|arg| visited[arg].clone()).collect_into(&mut arg_buf);

                let data = match node.cmd() {
                    ir::Command::Data => inputs[&node].clone(),
                    ir::Command::Add => add(&arg_buf)
                };

                visited.insert(node, data);
                arg_buf.clear();
            }
        }

        outputs.into_iter().map(|node| visited[&node].clone()).collect()
    }
}


#[derive(Clone)]
pub struct Tensor {
    buffer: Vec<f32>,
    shape: Shape,
}

impl Tensor {
    pub fn new(buffer: Vec<f32>, shape: Shape) -> Self {
        Tensor { buffer, shape }
    }
}


impl TensorPrimitive for Tensor {
    fn shape(&self) -> &Shape {
        &self.shape
    }
}


pub fn add(args: &[Tensor]) -> Tensor {



    Tensor {}
}