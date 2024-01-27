from neat.genomes.genome import Genome
from neat.innovation import InnovationRecord
from neat.parameters import SpeciationParams
from neat.species import Species, SpeciesInfo


def speciate(
    unspeciated: list[Genome],
    species_set: list[Species],
    params: SpeciationParams,
    innov_record: InnovationRecord,
) -> list[Species]:
    species_info = [s.info for s in species_set]
    unspeciated, new_species_info = select_new_representatives(
        unspeciated,
        species_info,
        params,
    )

    new_species_set: dict[int, Species] = {}
    for info in new_species_info:
        info.add_age()
        species = Species(info, params)
        new_species_set[species.id] = species

    for genome in unspeciated:
        candidates: list[tuple[int, float]] = []
        for species in new_species_set.values():
            distance = get_genomes_distance(species.representative, genome, params)
            if distance < params.compatibility_threshold:
                candidates.append((species.id, distance))

        if candidates:
            species_id = min(candidates, key=lambda x: x[1])[0]
            new_species_set[species_id].assign_genome(genome)
        else:
            new_species = get_new_species(genome, innov_record, params)
            new_species_set[new_species.id] = new_species

    return list(new_species_set.values())


def select_new_representatives(
    unspeciated: list[Genome],
    species_info: list[SpeciesInfo],
    params: SpeciationParams,
) -> tuple[list[Genome], list[SpeciesInfo]]:
    for info in species_info:
        representative = info.representative
        candidates: list[tuple[Genome, float]] = []
        for genome in unspeciated:
            distance = get_genomes_distance(representative, genome, params)
            candidates.append((genome, distance))

        best_candidate = min(candidates, key=lambda x: x[1])[0]
        info.representative = best_candidate
        unspeciated.remove(best_candidate)

    return unspeciated, species_info


def get_new_species(
    representative: Genome,
    innov_record: InnovationRecord,
    params: SpeciationParams,
) -> Species:
    info = SpeciesInfo(innov_record.get_species_id(), representative)
    return Species(info, params)


def get_genomes_distance(
    genome1: Genome,
    genome2: Genome,
    params: SpeciationParams,
) -> float:
    node_distance: float = 0.0
    disjoint_nodes: int = 0
    num_of_common_node_genes: int = 0

    if genome1.nodes or genome2.nodes:
        for node_id, node in genome1.nodes.items():
            if node2 := genome2.nodes.get(node_id):
                num_of_common_node_genes += 1
                node_distance += node.distance(node2)
            else:
                disjoint_nodes += 1

        disjoint_nodes += len(genome2.nodes) - num_of_common_node_genes
        max_nodes = max(len(genome1.nodes), len(genome2.nodes))
        node_distance = (
            node_distance + params.compatibility_disjoint_coefficient * disjoint_nodes
        ) / max_nodes

    link_distance = 0.0
    disjoint_links = 0
    num_of_common_link_genes = 0

    if genome1.links or genome2.links:
        for link_id, link in genome1.links.items():
            if link2 := genome2.links.get(link_id):
                num_of_common_link_genes += 1
                link_distance += link.distance(link2)
            else:
                disjoint_links += 1

        disjoint_links += len(genome2.links) - num_of_common_link_genes
        max_links = max(len(genome1.links), len(genome2.links))
        link_distance = (
            link_distance + params.compatibility_disjoint_coefficient * disjoint_links
        ) / max_links

    return node_distance + link_distance
