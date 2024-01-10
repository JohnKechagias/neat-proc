from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import neat.utils as utils
from neat.genes import Link, LinkTraits, Node, NodeTraits, NodeType
from neat.innovation import InnovationRecord
from neat.parameters import GenomeParams
from neat.types import LinkID, NodeID
from neat.utils import get_random_value


@dataclass
class Genome:
    params: GenomeParams

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
        if not nodes:
            nodes = {}

        if not links:
            links = {}

        self.id = id
        self.fitness = 0.0
        self.innov_record = innovation_record

        self.nodes: dict[NodeID, Node] = nodes
        self.links: dict[LinkID, Link] = links

        if not nodes and not links:
            self.initialize_default_genome()

    def __repr__(self) -> str:
        nodes_from_links = set()
        for l in self.links.values():
            nodes_from_links.add(l.in_node)
            nodes_from_links.add(l.out_node)

        return f"Genome(ID={self.id}, nodes={list(self.nodes.keys())}, nodes_from_links={list(nodes_from_links)}, links={list(self.links.keys())})"

    def initialize_default_genome(self):
        input_nodes: dict[int, Node] = {}
        output_nodes: dict[int, Node] = {}

        inputs = self.params.inputs
        outputs = self.params.outputs
        for node_id in range(inputs):
            traits = self.get_default_node_traits()
            node = Node(node_id, traits, NodeType.INPUT)
            input_nodes[node.id] = node

        for node_id in range(inputs, inputs + outputs):
            traits = self.get_default_node_traits()
            node = Node(node_id, traits, NodeType.OUTPUT)
            output_nodes[node.id] = node

        for input in input_nodes.values():
            for output in output_nodes.values():
                link = self.create_new_link(input.id, output.id)
                self.links[link.id] = link

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
        # print(f"Crossover: Combining Genome {primary_parent.id} with Genome {secondary_parent.id} producing Genome {id}")
        return Genome(id, self.innov_record, nodes, links)

    def mutate(self):
        if random.random() < self.params.node_deletion_chance:
            self.mutate_delete_node()

        if random.random() < self.params.node_addition_chance:
            self.mutate_add_node()

        if random.random() < self.params.link_addition_chance:
            self.mutate_add_link()

        if random.random() < self.params.link_deletion_chance:
            self.mutate_delete_link()

        if random.random() < self.params.link_toggle_chance:
            self.mutate_toggle_enable()

        for node in self.nodes.values():
            self.mutate_node(node)

        for link in self.links.values():
            self.mutate_link(link)

    def mutate_add_node(self):
        if not self.links:
            return

        link_to_split = get_random_value(self.links)
        link_to_split.disable()

        node = self.create_new_node(link_to_split.id)
        self.nodes[node.id] = node

        first_link = self.create_new_link(link_to_split.in_node, node.id)
        self.links[first_link.id] = first_link

        weight = link_to_split.weight
        second_link = self.create_new_link(node.id, link_to_split.out_node, weight)
        self.links[second_link.id] = second_link

    def mutate_delete_node(self):
        hidden_nodes = self.get_nodes_by_type(NodeType.HIDDEN)

        if not hidden_nodes:
            return

        node = random.choice(hidden_nodes)
        self.links = {
            k: l for k, l in self.links.items() if node.id not in l.simple_link
        }

        link_nodes = set()
        for i in self.links.values():
            link_nodes.add(i.in_node)
            link_nodes.add(i.out_node)

        self.nodes.pop(node.id)

    def mutate_add_link(self):
        in_nodes = [n for n in self.nodes.values() if n.node_type != NodeType.OUTPUT]
        out_nodes = [n for n in self.nodes.values() if n.node_type != NodeType.INPUT]

        in_node = random.choice(in_nodes)
        out_node = random.choice(out_nodes)

        if in_node == out_node:
            return

        for link in self.links.values():
            if link.simple_link == (in_node.id, out_node.id):
                return

        if creates_cycle(
            [l.simple_link for l in self.links.values()], (in_node.id, out_node.id)
        ):
            return

        link = self.create_new_link(in_node.id, out_node.id)

        assert link.id not in self.links
        self.links[link.id] = link

    def mutate_delete_link(self):
        if not self.links:
            return

        link = get_random_value(self.links)
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

    def mutate_node(self, node: Node):
        if random.random() < self.params.bias_mutation_chance:
            bias_change = random.gauss(0.0, self.params.bias_mutation_power)
            node.traits.bias = utils.clamp(
                node.traits.bias + bias_change,
                self.params.bias_min_value,
                self.params.bias_max_value,
            )

        if random.random() < self.params.bias_replace_chance:
            node.traits.bias = self.get_initial_bias()

        if random.random() < self.params.response_mutation_chance:
            response_change = random.gauss(0.0, self.params.response_mutation_power)
            node.traits.response = utils.clamp(
                node.traits.response + response_change,
                self.params.response_min_value,
                self.params.response_max_value,
            )

        if random.random() < self.params.response_replace_chance:
            node.traits.response = self.get_initial_response()

        # TODO add mutation of aggregation and activation functions.

    def mutate_link(self, link: Link):
        if link.frozen:
            return

        if not random.random() < self.params.weight_mutation_chance:
            return

        mutation_power = self.params.weight_mutation_power
        if random.random() < self.params.weight_severe_mutation_chance:
            mutation_power *= 2

        link.traits.weight = utils.clamp(
            link.weight + utils.randon_sign() * random.random() * mutation_power,
            self.params.weight_min_value,
            self.params.weight_max_value,
        )

    @property
    def size(self):
        """
        Returns genome 'complexity', taken to be
        (number of nodes, number of enabled connections)
        """
        num_enabled_connections = sum([1 for l in self.links.values() if l.enabled])
        return len(self.nodes), num_enabled_connections

    def create_new_node(
        self,
        link_to_split: LinkID,
        node_type: NodeType = NodeType.HIDDEN,
    ) -> Node:
        id = self.get_node_id(link_to_split)
        traits = self.get_default_node_traits()
        return Node(id, traits, node_type)

    def get_node_id(self, link_to_split: LinkID) -> int:
        return self.innov_record.get_node_id(link_to_split)

    def get_nodes_by_type(self, node_type: NodeType) -> list[Node]:
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def get_default_node_traits(self) -> NodeTraits:
        return NodeTraits(
            self.get_initial_bias(),
            self.get_initial_response(),
            self.params.aggregation_default,
            self.params.activation_default,
        )

    def get_initial_bias(self) -> float:
        bias = random.gauss(self.params.bias_init_mean, self.params.bias_init_stdev)
        return utils.clamp(bias, self.params.bias_min_value, self.params.bias_max_value)

    def get_initial_response(self) -> float:
        response = random.gauss(
            self.params.response_init_mean, self.params.response_init_stdev
        )
        return utils.clamp(
            response, self.params.response_min_value, self.params.response_max_value
        )

    def create_new_link(
        self, in_node: NodeID, out_node: NodeID, weight: float = 1
    ) -> Link:
        id = self.get_link_id(in_node, out_node)
        traits = LinkTraits(weight=weight)
        return Link(id, in_node, out_node, traits)

    def get_link_id(self, in_node: NodeID, out_node: NodeID) -> int:
        return self.innov_record.get_link_id(in_node, out_node)

    def reenable_random_link(self):
        for link in self.links.values():
            if not link.enabled:
                link.enable()
                break

    def reset_weights(self):
        for link in self.links.values():
            link.reset_weight()


def creates_cycle(connections: list[tuple[int, int]], test: tuple[int, int]):
    """
    Returns true if the addition of the 'test' connection would create a cycle,
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
