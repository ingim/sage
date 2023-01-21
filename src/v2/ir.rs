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
    Full(f32, Vec<usize>),
    Add,
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

    pub fn full(&mut self, scalar: f32) -> Node {
        self.create_node(Command::Add, [])
    }


    pub fn add(&mut self, x0: Node, x1: Node) -> Node {
        self.create_node(Command::Add, [x0, x1])
    }
}

