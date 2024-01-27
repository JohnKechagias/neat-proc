from __future__ import annotations

import random
from typing import Optional

from neat.genes import Link, Node, NodeType
from neat.innovation import InnovationRecord
from neat.parameters import GenomeParams
from neat.types import LinkID, NodeID, SLink
from neat.utils import get_random_value


class Genome:
    params: GenomeParams
    input_keys = set((0, 1))
    output_keys = set((2,))

    @classmethod
    def initialize_configuration(cls, params: GenomeParams):
        cls.params = params

    def __init__(
        self,
        id: int,
        innovation_record: InnovationRecord,
        nodes: Optional[dict[int, Node]] = None,
        links: Optional[dict[int, Link]] = None,
    ):
        self.parents = None
        if not nodes:
            nodes = {}

        if not links:
            links = {}

        self.id = id
        self.fitness = 0.0
        self.innov_record = innovation_record

        self.nodes: dict[NodeID, Node] = nodes
        self.links: dict[LinkID, Link] = links

        if not self.nodes and not self.links:
            self.initialize_default_genome()

    def __repr__(self) -> str:
        strings = []
        strings.append(f"\nGenome {self.id}")

        strings.append(f"Nodes:")
        for node in self.nodes.values():
            strings.append(f"    {node}")

        strings.append(f"Links:")
        for link in self.links.values():
            strings.append(f"    {link}")

        return "\n".join(strings)

    def initialize_default_genome(self):
        input_nodes: dict[int, Node] = {}
        output_nodes: dict[int, Node] = {}

        inputs = self.params.inputs
        outputs = self.params.outputs
        for node_id in range(inputs):
            node = self.create_new_node(node_id, NodeType.INPUT)
            input_nodes[node_id] = node

        for node_id in range(inputs, inputs + outputs):
            node = self.create_new_node(node_id, NodeType.OUTPUT)
            output_nodes[node_id] = node

        for input in input_nodes.keys():
            for output in output_nodes.keys():
                link_id = self.get_link_id(input, output)
                link = self.create_new_link(link_id, input, output)
                self.links[link_id] = link

        self.nodes = {**input_nodes, **output_nodes}

    def crossover(self, other: Genome) -> Genome:
        """Configure a new genome by crossover from two parent genomes."""

        primary_parent, secondary_parent = self, other
        if self.fitness < other.fitness:
            primary_parent, secondary_parent = other, self

        links = {}
        nodes = {}

        for key, link1 in primary_parent.links.items():
            if link2 := secondary_parent.links.get(key):
                # Homologous gene: combine genes from both parents.
                links[key] = link1.crossover(link2)
            else:
                # Excess or disjoint gene: copy from the fittest parent.
                links[key] = link1.copy()

        for key, node1 in primary_parent.nodes.items():
            if node2 := secondary_parent.nodes.get(key):
                # Homologous gene: combine genes from both parents.
                nodes[key] = node1.crossover(node2)
            else:
                # Extra gene: copy from the fittest parent
                nodes[key] = node1.copy()

        id = self.innov_record.get_genome_id()
        genome = Genome(id, self.innov_record, nodes, links)
        genome.parents = (self.id, other.id)
        return genome

    def mutate(self):
        if random.random() < self.params.node_addition_chance:
            self.mutate_add_node()

        if random.random() < self.params.node_deletion_chance:
            self.mutate_delete_node()

        if random.random() < self.params.link_addition_chance:
            self.mutate_add_link()

        if random.random() < self.params.link_deletion_chance:
            self.mutate_delete_link()

        if random.random() < self.params.link_toggle_chance:
            self.mutate_toggle_enable()

        for node in self.nodes.values():
            node.mutate(self.params)

        for link in self.links.values():
            link.mutate(self.params)

    def mutate_add_node(self):
        if not self.links:
            if self.params.alternative_structural_mutations:
                self.mutate_add_link()
            return

        link_to_split: Link = get_random_value(self.links)
        link_to_split.disable()

        node_id = self.get_node_id(link_to_split.id)
        node = self.create_new_node(node_id)
        self.nodes[node_id] = node

        flink_id = self.get_link_id(link_to_split.in_node, node_id)
        first_link = self.create_new_link(flink_id, link_to_split.in_node, node_id)
        first_link.weight = 1
        self.links[flink_id] = first_link

        slink_id = self.get_link_id(node_id, link_to_split.out_node)
        second_link = self.create_new_link(slink_id, node_id, link_to_split.out_node)
        second_link.weight = link_to_split.weight
        self.links[slink_id] = second_link

    def mutate_delete_node(self):
        hidden_nodes = self.get_nodes_by_type(NodeType.HIDDEN)

        if not hidden_nodes:
            return

        node_id = random.choice(hidden_nodes)
        self.links = {
            k: l for k, l in self.links.items() if node_id not in l.simple_link
        }
        self.nodes.pop(node_id)

    def mutate_add_link(self):
        in_nodes = [k for k, _ in self.nodes.items()]
        out_nodes = [k for k, n in self.nodes.items() if n.node_type != NodeType.INPUT]

        in_node = random.choice(in_nodes)
        out_node = random.choice(out_nodes)

        if in_node == out_node:
            return

        new_link = (in_node, out_node)
        for link in self.links.values():
            if link.simple_link == new_link:
                # TODO sometimes freezes, probably cycle?
                if self.params.alternative_structural_mutations:
                    link.enable()
                return

        if in_node in self.output_keys and out_node in self.output_keys:
            return

        simple_links = [l.simple_link for l in self.links.values()]
        if self.params.feed_forward and creates_cycle(simple_links, new_link):
            return

        link_id = self.get_link_id(in_node, out_node)
        link = self.create_new_link(link_id, in_node, out_node)
        assert link_id not in self.links
        self.links[link_id] = link

    def mutate_delete_link(self):
        if not self.links:
            return

        link: Link = get_random_value(self.links)
        self.links.pop(link.id)

    def mutate_toggle_enable(self):
        if not self.links:
            return

        link_to_toggle_enable: Link = get_random_value(self.links)

        if not link_to_toggle_enable.enabled:
            link_to_toggle_enable.enable()
            return

        # We need to make sure that another gene connects out of the in-node,
        # because if not, a section of network will break off and become isolated.
        in_node_id = link_to_toggle_enable.in_node
        sum_of_links_with_the_specific_in_node = 0

        for link in self.links.values():
            if link.in_node == in_node_id:
                sum_of_links_with_the_specific_in_node += 1

            if sum_of_links_with_the_specific_in_node >= 2:
                link_to_toggle_enable.disable()
                return

    @property
    def size(self):
        """
        Returns genome 'complexity', taken to be
        (number of nodes, number of enabled connections)
        """
        num_enabled_connections = sum([1 for l in self.links.values() if l.enabled])
        return len(self.nodes), num_enabled_connections

    def create_new_node(self, id: NodeID, ntype: NodeType = NodeType.HIDDEN) -> Node:
        bias, response, aggregator, activator = Node.get_default_args(self.params)
        return Node(id, ntype, bias, response, aggregator, activator)

    def get_node_id(self, link_to_split: LinkID) -> int:
        return self.innov_record.get_node_id(link_to_split)

    def get_nodes_by_type(self, node_type: NodeType) -> list[NodeID]:
        return [k for k, n in self.nodes.items() if n.node_type == node_type]

    def create_new_link(self, id: LinkID, in_node: NodeID, out_node: NodeID) -> Link:
        weight, enabled, frozen = Link.get_default_args(self.params)
        return Link(id, in_node, out_node, weight, enabled, frozen)

    def get_link_id(self, in_node: NodeID, out_node: NodeID) -> int:
        return self.innov_record.get_link_id(in_node, out_node)

    def reenable_random_link(self):
        for link in self.links.values():
            if not link.enabled:
                link.enable()
                break


def creates_cycle(connections: list[SLink], test: SLink) -> bool:
    """Returns true if the addition of the 'test' connection would create a cycle,
    assuming that no cycle already exists in the graph represented by 'connections'.
    """
    i, o = test
    if i == o:
        return True

    visited = {o}
    while True:
        num_added = 0
        for a, b in connections:
            if a in visited and b not in visited:
                if b == i:
                    return True

                visited.add(b)
                num_added += 1

        if num_added == 0:
            return False
