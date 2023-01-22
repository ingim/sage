use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use crate::v2::shape::Shape;

#[derive(Default)]
pub struct Graph {
    cmds: Vec<Command>,
    nodes: Vec<Node>,
    edges_in: HashMap<NodeId, Vec<NodeId>>,
    edges_out: HashMap<NodeId, Vec<NodeId>>,
    targets: Vec<NodeId>,
    id_counter: usize,
}

pub type NodeId = usize;

pub struct Node {
    shape: Shape,
    cmd: Command,
}

#[derive(Clone)]
pub enum Command {
    Data,
    Constant(f32),
    Map1(UnaryOperation),
    Map2(BinaryOperation),
    Map3(TernaryOperation),
}

#[derive(Copy, Clone)]
pub enum UnaryOperation {
    Copy,

    Abs,
    Neg,
    Recip,
    Log,
    Exp,
    Sqrt,
    Square,
    Sign,

    Ceil,
    Floor,
    Round,
    // trig
    Sin,
    Sinh,
    Cos,
    Cosh,
    Tan,
    Tanh,
    Asin,
    Asinh,
    Acos,
    Acosh,
    Atan,
    Atanh,

    // Status check
    IsNan,
    IsInf,

    // Logic
    Not,

}

#[derive(Copy, Clone)]
pub enum BinaryOperation {
    Add,
    Sub,
    Div,
    Mul,

    Mod,
    Pow,
    Min,
    Max,

    // Logic
    And,
    Or,
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
}

#[derive(Copy, Clone)]
pub enum TernaryOperation {
    Cond
}

impl Graph {
    pub fn new() -> Self {
        Graph::default()
    }

    pub fn edges_in(&self, node: NodeId) -> &[NodeId] {
        &self.edges_in[&node]
    }

    pub fn edges_out(&self, node: NodeId) -> &[NodeId] {
        &self.edges_out[&node]
    }

    pub fn targets(&self) -> &[NodeId] {
        &self.targets
    }

    pub fn add_target(&mut self, target: NodeId) {
        self.targets.push(target);
    }

    fn create_node<const N: usize>(&mut self, cmd: Command, args: [NodeId; N]) -> NodeId {
        let node_id = self.cmds.len();

        // analyze tensor shapes


        self.edges_in.insert(node_id, args.to_vec());

        args.into_iter().for_each(|arg| {
            self.edges_out.entry(arg).or_default().push(node_id);
        });

        self.cmds.push(cmd);
        node_id
    }

    pub fn cmd(&self, node: NodeId) -> &Command {
        &self.cmds[node.idx()]
    }

    pub fn data(&mut self) -> NodeId {
        self.create_node(Command::Data, [])
    }

    pub fn constant(&mut self, scalar: f32) -> NodeId {
        self.create_node(Command::Constant(scalar), [])
    }

    pub fn map1(&mut self, op: UnaryOperation, x: NodeId) -> NodeId {
        self.create_node(Command::Map1(op), [x])
    }

    pub fn map2(&mut self, op: BinaryOperation, x0: NodeId, x1: NodeId) -> NodeId {
        self.create_node(Command::Map2(op), [x0, x1])
    }

    pub fn map3(&mut self, op: TernaryOperation, x0: NodeId, x1: NodeId, x2: NodeId) -> NodeId {
        self.create_node(Command::Map3(op), [x0, x1, x2])
    }
}

