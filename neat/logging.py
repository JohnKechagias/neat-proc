from __future__ import annotations

import logging
import pickle
import sys
import time
from dataclasses import dataclass, field

from numba.core.utils import math

from neat.genomes import Genome
from neat.parameters import Parameters
from neat.species import Species
from neat.utils import mean, stdev

file_handler = logging.FileHandler(filename="training.log")
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    handlers=handlers,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class StatisticalData:
    stagnations: int = 0
    generation_times: list[float] = field(default_factory=lambda: [])
    average_fitnesses: list[float] = field(default_factory=lambda: [])
    standard_deviations: list[float] = field(default_factory=lambda: [])
    species_info: list = field(default_factory=lambda: [])

    def save_to_file(self, filepath: str = "stats.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def read_from_file(cls, filepath: str) -> StatisticalData:
        with open(filepath, "rb") as f:
            return pickle.load(f)


class Reporter:
    show_parameters: bool = False
    show_species_details: bool = True

    _header_width: int = 50

    def __init__(self):
        self.data = StatisticalData()

    def initialize_training(self, params: Parameters):
        if self.show_parameters:
            print(self.format_header(f"Parameters"))
            print(params)

        print(self.format_header(f"Starting Training"))

    def start_generation(self, generation_id: int):
        print(self.format_header(f"Running Generation {generation_id}"))
        self._generation_start_time = time.time()

    def end_generation(self, genomes: list[Genome], species_set: list[Species]):
        fitnesses = [genome.fitness for genome in genomes]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        print(f"Genomes Average fitness: {fit_mean:3.5f}")
        print(f"Genomes Standard deviation: {fit_std:3.5f}\n")
        print(self.format_header("Species"))
        self.data.average_fitnesses.append(fit_mean)
        self.data.standard_deviations.append(fit_std)

        elapsed_time = time.time() - self._generation_start_time
        self.data.generation_times.append(elapsed_time)
        mean_time = mean(self.data.generation_times)

        time_info = f"Generation time: {elapsed_time:.3f} sec ({mean_time:.3f} average)"
        pop_info = f"Population of {len(genomes)} members in {len(species_set)} species"
        print(time_info)
        print(pop_info)

        if not self.show_species_details:
            return

        print("   ID   age  size   fitness   stag")
        print("  ====  ===  ====  =========  ====")

        species_info = []
        for species in species_set:
            id = species.id
            age = species.age
            size = species.size
            f = f"{species.fitness:.3f}"
            stag = species.stagnant
            print(f"  {id:>4}  {age:>3}  {size:>4}  {f:>9}  {stag:>4}")
            species_info.append((id, age, size, f, stag))

        self.data.species_info.append(species_info)
        print(f"\nStagnations: {self.data.stagnations}")

    def best_genome(self, genome: Genome):
        print(
            f"\nBest genome is {genome.id} {genome.size}"
            f" with fitness {genome.fitness}\n"
        )

    def stagnant_species(self, species: int, size: int):
        self.data.stagnations += 1
        if self.show_species_details:
            print(
                f"\nSpecies {species} with {size} members is stangant. " "Removing it."
            )

    def format_header(self, text: str) -> str:
        num_of_chars_to_add = max(self._header_width - len(text) - 4, 0)
        num_of_left_chars = math.ceil(num_of_chars_to_add / 2)
        num_of_right_chars = num_of_chars_to_add // 2
        left_chars = "=" * num_of_left_chars
        right_chars = "=" * num_of_right_chars
        return f"+{left_chars} {text} {right_chars}+\n"

    def reset(self):
        self.data.save_to_file()
        self.data = StatisticalData()
