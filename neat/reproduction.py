import copy
import random
from math import ceil

from neat.genomes import Genome
from neat.logging import Reporter
from neat.parameters import ReproductionParams
from neat.species import Species
from neat.utils import mean


def filter_stagnant_species(
    species_set: list[Species],
    params: ReproductionParams,
    reporter: Reporter,
) -> list[Species]:
    is_stagnant = lambda s: s.stagnant > params.max_stagnation
    remaining_species: list[Species] = []
    for species in species_set:
        if is_stagnant(species):
            reporter.stagnant_species(species.id, species.size)
            continue

        remaining_species.append(species)

    if not remaining_species:
        raise RuntimeError(
            "There are no remaining species after evolution. "
            "Consider increasing the compatibility threshold."
        )

    return remaining_species


def reproduce(species_set: list[Species], params: ReproductionParams) -> list[Genome]:
    all_fitnesses: list[float] = []
    for species in species_set:
        all_fitnesses.extend(g.fitness for g in species.genomes)

    min_fitness = min(all_fitnesses)
    max_fitness = max(all_fitnesses)
    # Do not allow the fitness range to be zero, as we divide by it below.
    fitness_range = max(1.0, max_fitness - min_fitness)

    adjusted_fitnesses: list[float] = []
    for species in species_set:
        # Compute adjusted fitness.
        msf = mean([g.fitness for g in species.genomes])
        adjusted_fitness = (msf - min_fitness) / fitness_range
        adjusted_fitnesses.append(adjusted_fitness)
        species.fitness = adjusted_fitness

    avg_adjusted_fitness = mean(adjusted_fitnesses)
    print(f"Average adjusted fitness: {avg_adjusted_fitness:.3f}")
    offspring_per_species = compute_offspring_per_species(
        species_set,
        adjusted_fitnesses,
        params,
    )

    for species in species_set:
        species.sort_genomes_by_fitness()
        species.kill_worst(params.survival_rate, params.min_species_size)

    offspring: list[Genome] = []
    times_parents_are_the_same = 0
    for species, offspring_num in zip(species_set, offspring_per_species):
        if species.size >= params.elitism_threshold:
            elites = species.elites(params.elitism)
            offspring.extend(elites)
            offspring_num -= len(elites)

        if offspring_num <= 0:
            continue

        for _ in range(offspring_num):
            parent1 = random.choice(species.genomes)

            if random.random() < params.crossover_rate:
                if random.random() < params.inter_species_crossover_rate:
                    species_of_parent2 = random.choice(species_set)
                    parent2 = random.choice(species_of_parent2.genomes)
                else:
                    parent2 = random.choice(species.genomes)

                if parent1.id == parent2.id:
                    times_parents_are_the_same += 1

                # TODO might create cycles.
                child = parent1.crossover(parent2)
            else:
                id = parent1.innov_record.get_genome_id()
                # We need to copy nodes and links because if the parent is an elite it
                # will be transfered into the next generation and when we mutate him we
                # will also mutate the child by mistake.
                child = Genome(
                    id,
                    parent1.innov_record,
                    copy.copy(parent1.nodes),
                    copy.copy(parent1.links),
                )

            child.mutate()
            offspring.append(child)

    print(f"Times parents were the same {times_parents_are_the_same}")
    return offspring


def compute_offspring_per_species(
    species_set: list[Species],
    adjusted_fitnesses: list[float],
    params: ReproductionParams,
) -> list[int]:
    population = params.population
    min_species_size = params.min_species_size
    adj_fitness_sum = sum(adjusted_fitnesses)
    # Contains the number of genomes that each species must generate
    # to fill out the population.
    offspring_per_species: list[int] = []
    if adj_fitness_sum != 0:
        for fitness in adjusted_fitnesses:
            normalized_fitness = fitness / adj_fitness_sum
            genomes_spawn_count = ceil(population * normalized_fitness)
            genomes_spawn_count = max(min_species_size, genomes_spawn_count)
            offspring_per_species.append(genomes_spawn_count)
    else:
        # All members of all species have zero fitness.
        # Allocate each species an equal number of offspring.
        genomes_spawn_count = ceil(population / len(species_set))
        offspring_per_species = [genomes_spawn_count for _ in species_set]

    # Ensure that the species sizes sum is close to the population size.
    norm = population / sum(offspring_per_species)
    return [max(min_species_size, int(round(n * norm))) for n in offspring_per_species]
