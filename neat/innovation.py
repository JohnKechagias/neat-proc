from neat.types import GenomeID, LinkID, NodeID, SLink, SpeciesID


class InnovationRecord:
    def __init__(self, inputs: int, outputs: int):
        self.nodes_counter = inputs + outputs
        self.nodes_record: dict[LinkID, NodeID] = {}
        self.links_counter = 0
        self.links_record: dict[SLink, LinkID] = {}
        self.species_counter = 0
        self.genomes_counter = 0

    def get_node_id(self, link_to_split: LinkID) -> NodeID:
        node_id = self.nodes_record.get(link_to_split)

        if node_id is not None:
            return node_id

        node_id = self.nodes_counter
        self.nodes_record[link_to_split] = node_id
        self.nodes_counter += 1
        return node_id

    def get_link_id(self, in_node: NodeID, out_node: NodeID) -> LinkID:
        link = (in_node, out_node)
        link_id = self.links_record.get(link)

        if link_id is not None:
            return link_id

        link_id = self.links_counter
        self.links_record[link] = link_id
        self.links_counter += 1
        return link_id

    def get_species_id(self) -> SpeciesID:
        species_id = self.species_counter
        self.species_counter += 1
        return species_id

    def get_genome_id(self) -> GenomeID:
        genome_id = self.genomes_counter
        self.genomes_counter += 1
        return genome_id
