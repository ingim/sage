use std::collections::{HashMap, HashSet};

#[derive(Default)]
pub struct Graph {
    nodes: HashSet<Node>,
    edges_in: HashMap<Node, Vec<Node>>,
    edges_out: HashMap<Node, Vec<Node>>,

    id_counter: usize,
}

#[derive(Copy, Clone, Eq, Hash, PartialEq)]
pub struct Node {
    id: usize,
    cmd: Command,
}

#[derive(Copy, Clone, Eq, Hash, PartialEq)]
pub enum Command {
    Data,
    Add,
}


impl Node {
    fn new(id: usize, cmd: Command) -> Self {
        Node { id, cmd }
    }
}


impl Graph {
    pub fn new() -> Self {
        Graph::default()
    }

    fn create_node<const N: usize>(&mut self, cmd: Command, args: [Node; N]) -> Node {
        let node = Node::new(self.id_counter, cmd);
        self.id_counter += 1;

        self.nodes.insert(node);
        self.edges_in.insert(node, args.to_vec());

        for arg in args {
            self.edges_out.entry(arg).or_default().push(node);
        }

        node
    }


    pub fn data(&mut self) -> Node {
        self.create_node(Command::Data, [])
    }

    pub fn add(&mut self, x0: Node, x1: Node) -> Node {
        self.create_node(Command::Add, [x0, x1])
    }
}

