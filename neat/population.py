from typing import Callable, Optional

from neat.genomes.genome import Genome
from neat.innovation import InnovationRecord
from neat.logging import Reporter
from neat.parameters import Parameters
from neat.reproduction import reproduce
from neat.species import Species, SpeciesInfo
from neat.types import InputData


class Population:
    def __init__(self, params: Parameters):
        self.params = params
        Genome.initialize_configuration(params.genome)

        self.genomes: list[Genome]
        self.species: list[Species]
        self.generation: int
        self.prev_species_info: list[SpeciesInfo]
        self.innov_record: InnovationRecord

        self.reset()

    def run(
        self,
        fitness_func: Callable[[Genome, InputData], float],
        input: InputData,
        times: Optional[int] = None,
    ) -> Genome:
        best_genome: Optional[Genome] = None
        found_optimal_network = False
        Reporter.initialize_training(self.params)

        itterations = 0
        while not found_optimal_network:
            if times and itterations > times:
                break

            Reporter.start_generation(self.generation)
            self.evaluate(self.genomes, fitness_func, input)
            Reporter.end_generation(self.genomes, self.species)

            self.sort_genomes_by_fitness()
            best_genome = self.genomes[0]
            Reporter.best_genome(best_genome, 0)

            if best_genome.fitness >= self.params.evaluation.fitness_threshold:
                break

            self.genomes = reproduce(self.species, self.params.reproduction)
            self.species = self.speciate()
            self.generation += 1
            itterations += 1

        if best_genome is None:
            raise RuntimeError("Could not complete a full evolution cycle.")

        return best_genome

    def evaluate(
        self,
        genomes: list[Genome],
        fitness_func: Callable[[Genome, InputData], float],
        input: InputData,
    ):
        for genome in genomes:
            fitness = fitness_func(genome, input)
            genome.fitness = fitness

    def sort_genomes_by_fitness(self):
        self.genomes.sort(key=lambda g: g.fitness, reverse=True)

    def reset(self):
        self.genomes = []
        self.species = []
        self.generation = 0
        self.prev_species_info = []

        self.innov_record = InnovationRecord(
            self.params.genome.inputs,
            self.params.genome.outputs,
        )

        Reporter.reset()

        for _ in range(self.params.neat.population):
            self.genomes.append(self._get_new_genome())

        self.species = self.speciate()

    def speciate(self) -> list[Species]:
        new_species: list[Species] = []
        old_species_ids: list[int] = []

        for info in self.prev_species_info:
            info.add_age()
            new_species.append(Species(info))
            old_species_ids.append(info.id)

        for genome in self.genomes:
            found = False

            for species in new_species:
                if species.try_assign_genome(genome, self.params.speciation):
                    found = True
                    break

            if not found:
                species = self._get_new_species(genome)
                new_species.append(species)

        # Update representatives of species.
        for species in [s for s in new_species if s.id in old_species_ids]:
            # Remove the old representative from the species.
            species.genomes.remove(species.representative)

            # Don't include species with no members.
            if species.size == 0:
                new_species.remove(species)
                continue

            # Select the most fit genome as the representative.
            # TODO check if we should instead select the genome that is
            # closest to the old representative.
            species.sort_genomes_by_fitness()
            species.representative = species.genomes[0]

        self.prev_species_info = [species.info for species in new_species]
        return new_species

    def _get_new_genome(self) -> Genome:
        return Genome(self.innov_record.get_genome_id(), self.innov_record)

    def _get_new_species(self, representative: Genome) -> Species:
        info = SpeciesInfo(self.innov_record.get_species_id(), representative)
        return Species(info)
