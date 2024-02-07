import copy
from typing import Callable, Optional

from neat.genomes.genome import Genome
from neat.innovation import InnovationRecord
from neat.logging import Reporter
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
        fitness_func: Callable[[Genome, list[Genome]], float],
        times: Optional[int] = None,
    ) -> Genome:
        best_genome: Optional[Genome] = None
        found_optimal_network = False

        params = self.params
        population = params.reproduction.population
        genomes = [self.get_new_genome(self.innov_record) for _ in range(population)]
        species = speciate(genomes, [], params.speciation, self.innov_record)
        Reporter.initialize_training(params)

        iterations = 0
        while not found_optimal_network:
            if times and iterations > times:
                break

            Reporter.start_generation(self.generation)

            representatatives = get_representatives(species)
            for genome in genomes:
                genome.fitness = fitness_func(genome, representatatives)

            genomes.sort(key=lambda g: g.fitness, reverse=True)
            candidate_genome = genomes[0]

            if best_genome is None or candidate_genome.fitness > best_genome.fitness:
                best_genome = copy.deepcopy(candidate_genome)

            if best_genome.fitness >= params.evaluation.fitness_threshold:
                break

            species = filter_stagnant_species(species, params.reproduction)
            genomes = reproduce(species, params.reproduction)
            species = speciate(genomes, species, params.speciation, self.innov_record)

            Reporter.best_genome(best_genome, 0)
            Reporter.end_generation(genomes, species)
            self.generation += 1
            iterations += 1

        if best_genome is None:
            raise RuntimeError("Could not complete a full evolution cycle.")

        print(best_genome)
        return best_genome

    def reset(self):
        self.generation = 0
        self.innov_record = InnovationRecord(
            self.params.genome.inputs,
            self.params.genome.outputs,
        )

        Reporter.reset()

    @staticmethod
    def get_new_genome(innov_record: InnovationRecord) -> Genome:
        return Genome(innov_record.get_genome_id(), innov_record)
