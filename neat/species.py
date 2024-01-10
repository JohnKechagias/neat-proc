from __future__ import annotations

import random
from dataclasses import dataclass
from math import ceil
from typing import Optional

from neat.genomes.genome import Genome
from neat.parameters import SpeciationParams
from neat.speciation import are_genomes_compatible


@dataclass
class SpeciesInfo:
    id: int
    representative: Genome
    age: int = 0
    previous_fitness: float = 0
    stagnant: int = 0

    def add_age(self):
        self.age += 1


class Species:
    def __init__(self, info: SpeciesInfo):
        self.info = info
        # A sorted list with all the genomes in the Species.
        # The list is sorted from most fit to least fit genome.
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

    @stagnant.setter
    def stagnant(self, value: int):
        self.info.stagnant = value

    @property
    def fitness(self) -> float:
        return self.info.previous_fitness

    @fitness.setter
    def fitness(self, value: float):
        self.info.previous_fitness = value

    @property
    def representative(self) -> Genome:
        return self.info.representative

    @representative.setter
    def representative(self, value: Genome):
        self.info.representative = value

    def try_assign_genome(self, genome: Genome, params: SpeciationParams) -> bool:
        is_compatible = are_genomes_compatible(self.representative, genome, params)

        if is_compatible:
            self.genomes.append(genome)

        return is_compatible

    def force_assign_genome(self, genome: Genome):
        self.genomes.append(genome)

    def kill_worst(self, survival_rate: float):
        remaining = max(ceil(self.size * survival_rate), 1)
        self.genomes = self.genomes[:remaining]

    def sort_genomes_by_fitness(self):
        self.genomes.sort(key=lambda g: g.fitness, reverse=True)

    def update_adjusted_fitness(self) -> float:
        """Updates the species fitness value and returns it.

        Returns:
            The fitness of the species.
        """
        fitness_sum = 0.0
        for genome in self.genomes:
            fitness_sum += genome.fitness

        fitness = 0.0 if not self.genomes else fitness_sum / self.size**2

        if fitness < self.fitness:
            self.stagnant += 1
        else:
            self.stagnant = 0

        self.fitness = fitness
        return fitness

    def mate(self, parent2: Optional[Genome] = None) -> Genome:
        parent1 = random.choice(self.genomes)
        if parent2 is None:
            parent2 = random.choice(self.genomes)

        return parent1.crossover(parent2)

    def elites(self, count: int) -> list[Genome]:
        return self.genomes[:count]
