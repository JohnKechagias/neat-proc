import copy
from typing import Callable, Optional

from neat.genomes.genome import Genome
from neat.innovation import InnovationRecord
from neat.logging import Reporter
from neat.parameters import Parameters, SpeciationParams
from neat.reproduction import filter_stagnant_species, reproduce
from neat.species import Species, SpeciesInfo
from neat.speciation import get_genomes_distance
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
        params = self.params
        Reporter.initialize_training(params)

        itterations = 0
        while not found_optimal_network:
            if times and itterations > times:
                break

            Reporter.start_generation(self.generation)
            self.evaluate(self.genomes, fitness_func, input)
            self.sort_genomes_by_fitness()
            candidate_best_genome = self.genomes[0]

            if not best_genome or self.genomes[0].fitness > best_genome.fitness:
                best_genome = candidate_best_genome

            if best_genome.fitness >= params.evaluation.fitness_threshold:
                break

            species = filter_stagnant_species(self.species, params.reproduction)
            self.prev_species_info  = [s.info for s in species]

            self.genomes = reproduce(species, params.reproduction)
            self.species = self.speciate(params.speciation)

            Reporter.end_generation(self.genomes, self.species)
            Reporter.best_genome(best_genome, 0)
            self.generation += 1
            itterations += 1

        if best_genome is None:
            raise RuntimeError("Could not complete a full evolution cycle.")

        print(best_genome)

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

        self.species = self.speciate(self.params.speciation)

    def speciate(self, params: SpeciationParams) -> list[Species]:
        new_species_set: dict[int, Species] = {}
        
        unspeciated = copy.copy(self.genomes)
        unspeciated = self.select_new_representatives(unspeciated, params)
        
        for info in self.prev_species_info:
            info.add_age()
            species = Species(info)
            new_species_set[species.id] = species

        for genome in unspeciated:
            candidates: list[tuple[int, float]] = []
            for species in new_species_set.values():
                distance = species.get_distance(genome, params)
                if distance < params.compatibility_threshold:
                    candidates.append((species.id, distance))

            if candidates:
                species_id = min(candidates, key=lambda x: x[1])[0]
                new_species_set[species_id].force_assign_genome(genome)
            else:
                new_species = self._get_new_species(genome)
                new_species_set[new_species.id] = new_species
    
        self.prev_species_info = [species.info for species in new_species_set.values()]
        return list(new_species_set.values())

    def select_new_representatives(
        self,
        unspeciated: list[Genome],
        params: SpeciationParams,
    ) -> list[Genome]:
        for info in self.prev_species_info:
            representative = info.representative
            candidates: list[tuple[Genome, float]] = []
            for genome in unspeciated:
                distance = get_genomes_distance(representative, genome, params)
                candidates.append((genome, distance))

            best_candidate = min(candidates, key=lambda x: x[1])[0]
            info.representative = best_candidate
            unspeciated.remove(best_candidate)

        return unspeciated

    def _get_new_genome(self) -> Genome:
        return Genome(self.innov_record.get_genome_id(), self.innov_record)

    def _get_new_species(self, representative: Genome) -> Species:
        info = SpeciesInfo(self.innov_record.get_species_id(), representative)
        return Species(info)
