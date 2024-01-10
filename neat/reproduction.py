import copy
import random
from math import ceil

from neat.genomes import Genome
from neat.logging import LOGGER, Reporter
from neat.parameters import ReproductionParams
from neat.species import Species
from neat.utils import mean


def filter_stagnant_species(
    species_set: list[Species],
    params: ReproductionParams
) -> list[Species]:
    is_stagnant = lambda s: s.stagnant > params.max_stagnation
    remaining_species: list[Species] = []
    for species in species_set:
        if is_stagnant(species):
            Reporter.stagnant_species(species.id, species.size)
            continue

        remaining_species.append(species)

    if not remaining_species:
        LOGGER.warning(
            "There are no remaining species after evolution. "
            "Consider increasing the compatibility threshold."
        )

    return remaining_species


def reproduce(species_set: list[Species], params: ReproductionParams) -> list[Genome]:
    offspring: list[Genome] = []
    for species in species_set:
        species.sort_genomes_by_fitness()
        species.kill_worst(params.survival_rate)

        if species.size >= params.elitism_threshold:
            offspring.extend(species.elites(params.elitism))

    genomes_to_spawn = params.population - len(offspring)
    offspring_per_species = compute_offspring_num(species_set, genomes_to_spawn)

    for species, offspring_num in zip(species_set, offspring_per_species):
        for _ in range(offspring_num):
            parent1 = random.choice(species.genomes)

            if random.random() < params.crossover_rate:
                parent2 = random.choice(species.genomes)

                if random.random() < params.inter_species_crossover_rate:
                    species_of_parent2 = random.choice(species_set)
                    parent2 = random.choice(species_of_parent2.genomes)

                child = parent1.crossover(parent2)
            else:
                id = parent1.innov_record.get_genome_id()
                # We need to copy nodes and links because if the parent is an elite it will be transfered
                # into the next generation and when we mutate him we will also mutate the child by mistake.
                child = Genome(
                    id,
                    parent1.innov_record,
                    copy.copy(parent1.nodes),
                    copy.copy(parent1.links),
                )
                child.mutate()

            offspring.append(child)

    return offspring


def compute_offspring_num(species_set: list[Species], population: int) -> list[int]:
    adjusted_fitnesses = [f.update_adjusted_fitness() for f in species_set]
    adj_fitness_sum = sum(adjusted_fitnesses)
    avg_adjusted_fitness = mean(adjusted_fitnesses)
    LOGGER.info(f"Average adjusted fitness: {avg_adjusted_fitness:.3f}")

    # Contains the number of genomes that each species must generate
    # to fill out the population.
    offsprings_per_species: list[int] = []
    if adj_fitness_sum != 0:
        for fitness in adjusted_fitnesses:
            normalized_fitness = fitness / adj_fitness_sum
            genomes_spawn_count = ceil(population * normalized_fitness)
            offsprings_per_species.append(genomes_spawn_count)
    else:
        # All members of all species have zero fitness.
        # Allocate each species an equal number of offspring.
        genomes_spawn_count = ceil(population / len(species_set))
        offsprings_per_species = [genomes_spawn_count for _ in species_set]

    # Ensure that the species sizes sum to population size.
    extra_genomes_generated = sum(offsprings_per_species) - population
    for i in range(extra_genomes_generated):
        offsprings_per_species[i % len(species_set)] -= 1

    return offsprings_per_species
