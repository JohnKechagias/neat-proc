from neat.genomes.genome import Genome
from neat.parameters import SpeciationParams


def are_genomes_compatible(
    genome1: Genome,
    genome2: Genome,
    params: SpeciationParams,
) -> bool:
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

    genomes_distance = node_distance + link_distance
    return genomes_distance < params.compatibility_threshold
