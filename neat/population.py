import copy
import pickle
from typing import Callable, Optional

from neat.genomes.genome import Genome
from neat.innovation import InnovationRecord
from neat.logging import Reporter, StatisticalData
from neat.parameters import Parameters
from neat.reproduction import filter_stagnant_species, reproduce
from neat.speciation import get_representatives, speciate


class Population:
    def __init__(self, params: Parameters):
        self.params = params
        self.generation: int
        self.innov_record: InnovationRecord
        self.reset()
        Genome.initialize_configuration(params.genome)

    def run(
        self,
        fitness_func: Callable[[list[Genome], list[Genome]], None],
        times: Optional[int] = None,
    ) -> tuple[Genome, StatisticalData]:
        best_genome: Optional[Genome] = None
        found_optimal_network = False

        params = self.params
        population = params.reproduction.population
        genomes = [self.get_new_genome(self.innov_record) for _ in range(population)]
        species = speciate(genomes, [], params.speciation, self.innov_record)
        reporter = Reporter()
        reporter.initialize_training(params)

        iterations = 0
        while not found_optimal_network:
            if times is not None and iterations > times:
                break

            reporter.start_generation(self.generation)

            representatatives = get_representatives(species)
            fitness_func(genomes, representatatives)
            genomes.sort(key=lambda g: g.fitness, reverse=True)
            candidate_genome = genomes[0]

            if best_genome is None or candidate_genome.fitness > best_genome.fitness:
                best_genome = copy.deepcopy(candidate_genome)

            reporter.best_genome(best_genome)

            with open(f"generation_{self.generation}_winner.pkl", "wb") as f:
                pickle.dump(best_genome, f)

            if best_genome.fitness >= params.evaluation.fitness_threshold:
                break

            species = filter_stagnant_species(species, params.reproduction, reporter)
            genomes = reproduce(species, params.reproduction)
            species = speciate(genomes, species, params.speciation, self.innov_record)

            reporter.end_generation(genomes, species)
            reporter.data.save_to_file(f"generation_{self.generation}_stats.pkl")
            self.generation += 1
            iterations += 1

        reporter.data.save_to_file()
        if best_genome is None:
            raise RuntimeError("Could not complete a full evolution cycle.")

        print(best_genome)
        return best_genome, reporter.data

    def reset(self):
        self.generation = 0
        self.innov_record = InnovationRecord(
            self.params.genome.inputs,
            self.params.genome.outputs,
        )

    @staticmethod
    def get_new_genome(innov_record: InnovationRecord) -> Genome:
        return Genome(innov_record.get_genome_id(), innov_record)
