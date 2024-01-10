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
    logger: logging.Logger = LOGGER
    show_parameters: bool = False
    show_species_details: bool = True

    _header_width: int = 50
    _num_of_extinctions: int = 0
    _generation_times: list[float] = []

    @classmethod
    def initialize_training(cls, params: Parameters):
        if cls.show_parameters:
            cls.logger.info(cls.format_header(f"Parameters"))
            cls.logger.info(params)

        cls.logger.info(cls.format_header(f"Starting Training"))

    @classmethod
    def start_generation(cls, generation_id: int):
        header = cls.format_header(f"Running Generation {generation_id}")
        cls.logger.info(header)
        cls._generation_start_time = time.time()

    @classmethod
    def end_generation(cls, genomes: list[Genome], species_set: list[Species]):
        fitnesses = [genome.fitness for genome in genomes]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        cls.logger.info(f"Genomes Average fitness: {fit_mean:3.5f}")
        cls.logger.info(f"Genomes Standard deviation: {fit_std:3.5f}\n")

        cls.logger.info(cls.format_header("Species"))
        elapsed_time = time.time() - cls._generation_start_time
        cls._generation_times.append(elapsed_time)
        mean_time = mean(cls._generation_times)

        time_info = f"Generation time: {elapsed_time:.3f} sec ({mean_time:.3f} average)"
        cls.logger.info(time_info)
        pop_info = f"Population of {len(genomes)} members in {len(species_set)} species"
        cls.logger.info(pop_info)

        if not cls.show_species_details:
            return

        cls.logger.info("   ID   age  size   fitness   stag")
        cls.logger.info("  ====  ===  ====  =========  ====")

        for species in species_set:
            id = species.id
            age = species.age
            size = species.size
            f = f"{species.fitness:.3f}"
            stag = species.stagnant
            cls.logger.info(f"  {id:>4}  {age:>3}  {size:>4}  {f:>9}  {stag:>4}")

        cls.logger.info("")

    @classmethod
    def best_genome(cls, genome: Genome, species: int):
        cls.logger.info("")
        cls.logger.info(f"Best genome is {genome.id} from species {species} with complexity {genome.size}")
        cls.logger.info("")

    @classmethod
    def stagnant_species(cls, species: int, size: int):
        if cls.show_species_details:
            cls.logger.info(
                f"\nSpecies {species} with {size} members is stangant. "
                "Removing it."
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
