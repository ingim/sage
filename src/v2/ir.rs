use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

#[derive(Default)]
pub struct Graph {
    cmds: Vec<Command>,
    edges_in: HashMap<Node, Vec<Node>>,
    edges_out: HashMap<Node, Vec<Node>>,
    targets: Vec<Node>,
    id_counter: usize,
}

#[derive(Copy, Clone)]
pub struct Node {
    idx: usize,
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
    Id,

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

impl Node {
    fn new(id: usize) -> Self {
        Node { idx: id }
    }

    pub fn idx(&self) -> usize {
        self.idx
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx
    }
}

impl Eq for Node {}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.idx.hash(state);
    }
}

impl Graph {
    pub fn new() -> Self {
        Graph::default()
    }

    pub fn edges_in(&self, node: Node) -> &[Node] {
        &self.edges_in[&node]
    }

    pub fn edges_out(&self, node: Node) -> &[Node] {
        &self.edges_out[&node]
    }

    pub fn targets(&self) -> &[Node] {
        &self.targets
    }

    pub fn add_target(&mut self, target: Node) {
        self.targets.push(target);
    }

    fn create_node<const N: usize>(&mut self, cmd: Command, args: [Node; N]) -> Node {
        let node = Node::new(self.cmds.len());

        self.edges_in.insert(node, args.to_vec());

        args.into_iter().for_each(|arg| {
            self.edges_out.entry(arg).or_default().push(node);
        });

        self.cmds.push(cmd);
        node
    }

    pub fn cmd(&self, node: Node) -> &Command {
        &self.cmds[node.idx()]
    }

    pub fn data(&mut self) -> Node {
        self.create_node(Command::Data, [])
    }

    pub fn constant(&mut self, scalar: f32) -> Node {
        self.create_node(Command::Constant(scalar), [])
    }

    pub fn map1(&mut self, op: UnaryOperation, x: Node) -> Node {
        self.create_node(Command::Map1(op), [x])
    }

    pub fn map2(&mut self, op: BinaryOperation, x0: Node, x1: Node) -> Node {
        self.create_node(Command::Map2(op), [x0, x1])
    }

    pub fn map3(&mut self, op: TernaryOperation, x0: Node, x1: Node, x2: Node) -> Node {
        self.create_node(Command::Map3(op), [x0, x1, x2])
    }
}

