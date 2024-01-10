from __future__ import annotations

import numpy as np

from neat.genomes.genome import Genome

from .network import Network


class FeedForwardNetwork(Network):
    @staticmethod
    def from_genome(genome: Genome) -> FeedForwardNetwork:
        return FeedForwardNetwork(genome.id, genome.nodes, genome.links)

    def activate(self, input: np.ndarray) -> list[float]:
        if len(input) != len(self.inputs):
            raise RuntimeError(
                f"Expected {len(self.inputs)} inputs but got {len(input)}."
            )

        for node, value in zip(self.inputs, input):
            self.node_values[node] = value

        for node_id in self.graph.toposort():
            evaluator = self.node_evaluators[node_id]
            output = evaluator(self.node_inputs[node_id])
            self.node_values[node_id] = output

            for neighbor, weight in self.graph.get_neighbors(node_id).items():
                self.node_inputs[neighbor].append(output * weight)

        return [v for node, v in self.node_values.items() if node in self.outputs]
