from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

from neat.genomes.genome import Genome
from neat.parameters import SpeciationParams


@dataclass
class SpeciesInfo:
    id: int
    representative: Genome
    age: int = 0
    fitness_history: list[float] = field(default_factory=lambda: [0])
    stagnant: int = 0

    def add_age(self):
        self.age += 1


class Species:
    def __init__(self, info: SpeciesInfo, params: SpeciationParams):
        self.info = info
        self.max_fitness_history_size = params.max_stagnation
        self.genomes = [info.representative]

    def __le__(self, other: Species) -> bool:
        return self.fitness <= other.fitness

    @property
    def size(self) -> int:
        return len(self.genomes)

    @property
    def age(self) -> int:
        return self.info.age

    @property
    def id(self) -> int:
        return self.info.id

    @property
    def stagnant(self) -> int:
        return self.info.stagnant

    @property
    def fitness(self) -> float:
        return self.info.fitness_history[-1]

    @fitness.setter
    def fitness(self, value: float):
        self.info.fitness_history.append(value)

        if len(self.info.fitness_history) > self.max_fitness_history_size:
            self.info.fitness_history.pop(0)

        if self.info.fitness_history[-1] - self.info.fitness_history[-2] < 0:
            self.info.stagnant += 1

    @property
    def fitness_history(self) -> list[float]:
        return self.info.fitness_history

    @property
    def representative(self) -> Genome:
        return self.info.representative

    @representative.setter
    def representative(self, value: Genome):
        self.info.representative = value

    def assign_genome(self, genome: Genome):
        self.genomes.append(genome)

    def kill_worst(self, survival_rate: float, min_size: int):
        remaining = max(ceil(self.size * survival_rate), min_size)
        self.genomes = self.genomes[:remaining]

    def sort_genomes_by_fitness(self):
        self.genomes.sort(key=lambda g: g.fitness, reverse=True)

    def elites(self, count: int) -> list[Genome]:
        return self.genomes[:count]
