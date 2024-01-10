from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from neat.genomes.genome import Genome
from neat.types import NodeID


class Network(ABC):
    def __init__(
        self,
        id: int,
        inputs: list[NodeID],
        outputs: list[NodeID],
        evaluators: list[tuple[NodeID, Callable, list[tuple[NodeID, float]]]]
    ):
        self.id = id
        self.inputs = inputs
        self.outputs = outputs
        self.evaluators = evaluators
        self.values = {node_id: 0.0 for node_id in inputs + outputs}

    @staticmethod
    @abstractmethod
    def from_genome(genome: Genome) -> Network:
        """Receives a genome and returns its phenotype (Network)."""

    @abstractmethod
    def activate(self, input: np.ndarray) -> list[float]:
        """Passes the given input to the network."""
