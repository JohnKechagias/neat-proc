from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from neat.logging import LOGGER
from neat.types import NodeID

GLink = tuple[NodeID, NodeID]


class CycleFoundError(Exception):
    """Raised whenever a graph cycle is found when it should not.
    For example when trying to topologically sort the graph."""


class NodeNotFoundError(Exception):
    """Raised when trying to access a graph node that does not exist."""


class LinkNotFoundError(Exception):
    """Raised when trying to access a graph link that does not exist."""


@dataclass
class GNode:
    id: NodeID
    neighbors: dict[NodeID, float] = field(default_factory=dict)

    def add_neighbor(self, neighbor: NodeID, weight: float):
        self.neighbors[neighbor] = weight

    def remove_neighbor(self, neighbor: NodeID):
        self.neighbors.pop(neighbor)


class Graph:
    def __init__(self):
        self.graph: dict[NodeID, GNode] = {}
        self.in_degrees: dict[NodeID, int] = {}

    def get_neighbors(self, node: NodeID) -> dict[NodeID, float]:
        return self.graph[node].neighbors

    def add_node(self, node: NodeID):
        if node in self.graph:
            LOGGER.info(f"Attempted to add already existing node '{node}'.")
            return

        self.graph[node] = GNode(node)
        self.in_degrees[node] = 0

    def remove_node(self, node: NodeID):
        if node not in self.graph:
            raise NodeNotFoundError(f"Node '{node}' does not exist.")

        for neighbor in self.graph[node].neighbors:
            self.in_degrees[neighbor] -= 1

        self.graph.pop(node)
        self.in_degrees.pop(node)

    def add_link(self, input: NodeID, output: NodeID, weight: float):
        if input not in self.graph:
            raise NodeNotFoundError(f"The input node {input} is not in the graph.")

        if output not in self.graph:
            raise NodeNotFoundError(f"The output node {output} is not in the graph.")

        self.graph[input].add_neighbor(output, weight)
        self.in_degrees[output] += 1

    def remove_link(self, input: NodeID, output: NodeID):
        if output not in self.graph[input].neighbors:
            raise LinkNotFoundError(f"Link '{input} -> {output}' does not exist.")

        self.graph[input].remove_neighbor(output)
        self.in_degrees[output] -= 1

    def edit_link_weight(self, input: NodeID, output: NodeID, weight: float):
        self.graph[input].add_neighbor(output, weight)

    def toposort(self, stack: Optional[list[NodeID]] = None) -> list[NodeID]:
        """Sorts the nodes in topological order. That means that, for any given
        node, nodes that need to be activated before it are on the left of it
        and nodes that need to be activate after it are on the right of it.

        Based on Kahn's algorithm.

        Returns:
            The list of sorted node layers (input to output layer).
        """
        if stack is None:
            stack = []

        in_degree: dict[NodeID, int] = self.in_degrees.copy()
        ordered: list[NodeID] = []

        for node, in_degrees in in_degree.items():
            if in_degrees == 0:
                stack.append(node)

        num_of_visited_nodes = 0
        while stack:
            node = stack.pop()
            ordered.append(node)

            for neighbor in self.graph[node].neighbors:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    stack.append(neighbor)

            num_of_visited_nodes += 1

        if num_of_visited_nodes != len(self.graph):
            print(f"Current path: {stack}")
            print(f"All nodes: {self.graph.keys()}")
            print(f"In degrees: {in_degree}")
            print(f"Original in degrees: {self.in_degrees}")
            print(f"Num of visited nodes: {num_of_visited_nodes}")
            print(f"Connections: {self.graph.values()}")
            print(f"Ordered list: {ordered}")
            raise CycleFoundError("Tried to topologically sort a cyclic graph.")

        return ordered


def get_required_hidden_nodes(
    inputs: list[NodeID],
    outputs: list[NodeID],
    hidden_nodes: list[NodeID],
    links: list[GLink],
) -> list[NodeID]:
    """Check to see if a node is required for computing the output of the
    network.

    A hidden node h in a NN is required if the following hold:
        a) there is a path from h to an output node
        b) there is a path from an input node to h

    Shortcuts can be taken if there is a path from h1 to h2 and h1 has been marked
    as required.

    Args:
        inputs (list): The keys of input nodes.
        biases (list): The keys of bias nodes.
        outputs (list): The keys of output nodes.
        connections (list): A list of tuples that specify the input and output
            node keys of each enabled connection.
        nodes (list): The keys of all nodes in the network.

    Returns:
        set: The set of nodes required for computing the output of the network.
    """
    non_hidden_nodes = set(inputs + outputs)
    required = set()

    for h in hidden_nodes:
        if h in required:
            continue

        # if the node hasn't already been marked as required.
        path_to_output = find_path([h], outputs + list(required), links)
        path_from_input = find_path(inputs + list(required), [h], links)

        if path_to_output and path_from_input:
            # Add hidden nodes along the path found.
            for node in path_from_input + path_to_output:
                if node not in non_hidden_nodes:
                    required.add(node)

    return list(required)


def find_path(
    sources: list[NodeID], goals: list[NodeID], links: list[GLink]
) -> list[NodeID]:
    """Try to find a path between the any of the start nodes and any of
    the goal nodes.

    Returns:
        A list of each node along the discovered path.
    """
    visited: set[NodeID] = set()
    queue = deque()

    for node in sources:
        queue.appendleft([node])

    while queue:
        path: list[NodeID] = queue.pop()
        head = path[-1]
        visited.add(head)

        neighbours = [o for (i, o) in links if i == head]
        for neighbour in neighbours:
            if neighbour in goals:
                return path + [neighbour]
            elif neighbour not in visited:
                queue.appendleft(path + [neighbour])

    return []
