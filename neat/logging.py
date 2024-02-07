import logging
import sys
import time

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


class Reporter:
    show_parameters: bool = False
    show_species_details: bool = True

    _header_width: int = 50
    _num_of_extinctions: int = 0
    _generation_times: list[float] = []
    _stagnation = 0

    @classmethod
    def initialize_training(cls, params: Parameters):
        if cls.show_parameters:
            print(cls.format_header(f"Parameters"))
            print(params)

        print(cls.format_header(f"Starting Training"))

    @classmethod
    def start_generation(cls, generation_id: int):
        print(cls.format_header(f"Running Generation {generation_id}"))
        cls._generation_start_time = time.time()

    @classmethod
    def end_generation(cls, genomes: list[Genome], species_set: list[Species]):
        fitnesses = [genome.fitness for genome in genomes]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        print(f"Genomes Average fitness: {fit_mean:3.5f}")
        print(f"Genomes Standard deviation: {fit_std:3.5f}\n")
        print(cls.format_header("Species"))

        elapsed_time = time.time() - cls._generation_start_time
        cls._generation_times.append(elapsed_time)
        mean_time = mean(cls._generation_times)

        time_info = f"Generation time: {elapsed_time:.3f} sec ({mean_time:.3f} average)"
        pop_info = f"Population of {len(genomes)} members in {len(species_set)} species"
        print(time_info)
        print(pop_info)

        if not cls.show_species_details:
            return

        print("   ID   age  size   fitness   stag")
        print("  ====  ===  ====  =========  ====")

        for species in species_set:
            id = species.id
            age = species.age
            size = species.size
            f = f"{species.fitness:.3f}"
            stag = species.stagnant
            print(f"  {id:>4}  {age:>3}  {size:>4}  {f:>9}  {stag:>4}")

        print(f"\nStagnations: {cls._stagnation}")

    @classmethod
    def best_genome(cls, genome: Genome):
        print(
            f"\nBest genome is {genome.id} {genome.size}"
            f" with fitness {genome.fitness}\n"
        )

    @classmethod
    def stagnant_species(cls, species: int, size: int):
        cls._stagnation += 1
        if cls.show_species_details:
            print(
                f"\nSpecies {species} with {size} members is stangant. " "Removing it."
            )

    @classmethod
    def format_header(cls, text: str) -> str:
        num_of_chars_to_add = max(cls._header_width - len(text) - 4, 0)
        num_of_left_chars = math.ceil(num_of_chars_to_add / 2)
        num_of_right_chars = num_of_chars_to_add // 2
        left_chars = "=" * num_of_left_chars
        right_chars = "=" * num_of_right_chars
        return f"+{left_chars} {text} {right_chars}+\n"

    @classmethod
    def reset(cls):
        cls._num_of_extinctions = 0
        cls._generation_times.clear()
