from __future__ import annotations

import numpy as np
from neat.genes import NodeType

from neat.genomes.genome import Genome

from .network import Network
from .utils import get_feed_forward_layers


class FeedForwardNetwork(Network):
    @staticmethod
    def from_genome(genome: Genome) -> FeedForwardNetwork:
        input_nodes = genome.get_nodes_by_type(NodeType.INPUT)
        output_nodes = genome.get_nodes_by_type(NodeType.OUTPUT)
        links = [l for l in genome.links.values() if l.enabled]
        slinks = [l.simple_link for l in links]

        layers = get_feed_forward_layers(input_nodes, output_nodes, slinks)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                for link in [l for l in links if l.out_node == node]:
                    inputs.append((link.in_node, link.weight))

                node_evals.append((node, genome.nodes[node].get_evaluator(), inputs))

        return FeedForwardNetwork(genome.id, input_nodes, output_nodes, node_evals)

    def activate(self, input: np.ndarray) -> list[float]:
        if len(input) != len(self.inputs):
            raise RuntimeError(
                f"Expected {len(self.inputs)} inputs but got {len(input)}."
            )

        for node, value in zip(self.inputs, input):
            self.values[node] = value

        for node, evaluator, links in self.evaluators:
            node_inputs = []
            for neighbor, weight in links:
                node_inputs.append(self.values[neighbor] * weight)

            self.values[node] = evaluator(node_inputs)

        return [self.values[node] for node in self.outputs]
