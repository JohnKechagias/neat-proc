from neat.types import NodeID, SLink


def required_for_output(
    inputs: list[NodeID],
    outputs: list[NodeID],
    links: list[SLink],
) -> set[NodeID]:
    """Collect the nodes whose state is required to compute the final network output(s).
    It is assumed that the input identifier set and the node identifier set are disjoint.
    By convention, the output node ids are always the same as the output index.

    Returns: The set or required nodes.
    """
    assert not set(inputs).intersection(outputs)

    required = set(outputs)
    s = set(outputs)
    while True:
        # Find nodes not in s whose output is consumed by a node in s.
        t = set(a for (a, b) in links if b in s and a not in s)

        if not t:
            break

        layer_nodes = set(x for x in t if x not in inputs)
        if not layer_nodes:
            break

        required = required.union(layer_nodes)
        s = s.union(t)

    return required


def get_feed_forward_layers(
    inputs: list[NodeID],
    outputs: list[NodeID],
    links: list[SLink],
) -> list[set[NodeID]]:
    """Collect the layers whose members can be evaluated in parallel in a feed-forward network.

    Returns:
        A list of layers, with each layer consisting of a set of node identifiers.
        Note that the returned layers do not contain nodes whose output is ultimately
        never used to compute the final network output.
    """

    required = required_for_output(inputs, outputs, links)

    layers: list[set[NodeID]] = []
    s = set(inputs)
    while True:
        # Find candidate nodes c for the next layer.  These nodes should connect
        # a node in s to a node not in s.
        c = set(b for (a, b) in links if a in s and b not in s)
        # Keep only the used nodes whose entire input set is contained in s.
        t: set[NodeID] = set()
        for n in c:
            if n in required and all(a in s for (a, b) in links if b == n):
                t.add(n)

        if not t:
            break

        layers.append(t)
        s = s.union(t)

    return layers
