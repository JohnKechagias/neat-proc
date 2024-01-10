from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from neat.genes import Link, Node, NodeType
from neat.genomes.genome import Genome
from neat.types import LinkID, NodeID

from .graph import Graph, get_required_nodes


class Network(ABC):
    def __init__(self, id: int, nodes: dict[NodeID, Node], links: dict[LinkID, Link]):
        self.id = id
        self.graph = Graph()

        inputs = [k for k, n in nodes.items() if n.node_type == NodeType.INPUT]
        outputs = [k for k, n in nodes.items() if n.node_type == NodeType.OUTPUT]
        hidden = [k for k, n in nodes.items() if n.node_type == NodeType.HIDDEN]
        glinks = [l.simple_link for l in links.values() if l.enabled]
        self.inputs = inputs
        self.outputs = set(outputs)

        req_nodes = get_required_nodes(inputs, outputs, hidden, glinks)
        req_links = [
            l
            for l in links.values()
            if l.enabled and l.in_node in req_nodes and l.out_node in req_nodes
        ]

        r = [n for n in nodes.values() if n.id in req_nodes]
        self.node_inputs = {node_id: [] for node_id in req_nodes}
        self.node_values = {node_id: 0.0 for node_id in req_nodes}
        self.node_evaluators = {n.id: n.get_evaluator() for n in r}

        for node_id in req_nodes:
            self.graph.add_node(node_id)

        for link in req_links:
            self.graph.add_link(link.in_node, link.out_node, link.weight)

    @staticmethod
    @abstractmethod
    def from_genome(genome: Genome) -> Network:
        """Receives a genome and returns its phenotype (Network)."""

    @abstractmethod
    def activate(self, input: np.ndarray) -> list[float]:
        """Passes the given input to the network."""
