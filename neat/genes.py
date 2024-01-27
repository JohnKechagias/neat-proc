from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from inspect import get_annotations
from typing import Any, Callable, ClassVar, Self

from neat.activations import ActivationFuncs, get_activation_func
from neat.aggregations import AggregationFuncs, get_aggregation_func
from neat.parameters import GenomeParams
from neat.types import LinkID, NodeID, SLink
from neat.utils import clamp


class NodeType(StrEnum):
    INPUT = auto()
    HIDDEN = auto()
    OUTPUT = auto()


class Gene:
    def __init__(self, id: int):
        self.id = id

    _excluded_attrs_from_mutation: ClassVar[list[str]] = [
        "id",
        "node_type",
        "in_node",
        "out_node",
    ]

    def __repr__(self) -> str:
        attrs_strings: list[str] = []

        for attr in get_annotations(self.__class__, eval_str=True).keys():
            value = getattr(self, attr)

            attrs_strings.append(f"{attr}={value}")

        content = ", ".join(attrs_strings)
        return f"{self.__class__.__name__}({content})"

    def distance(self, other: Self) -> float:
        """Returns the distance between the genes. Distance
        measures how different the genes are."""
        distance = 0.0

        for attr, attr_type in get_annotations(self.__class__, eval_str=True).items():
            if attr in self._excluded_attrs_from_mutation:
                continue

            value = getattr(self, attr)
            other_value = getattr(other, attr)

            if issubclass(attr_type, (bool, Enum)):
                if value != other_value:
                    distance += 1.0
            elif issubclass(attr_type, (int, float)):
                distance += abs(value - other_value)
            else:
                raise NotImplementedError("Mutation attr type is not implemented.")

        return distance

    def crossover(self, other: Self) -> Self:
        """Return a new Gene that is a combination of the two. It
        inherits values from both parents with equal probability."""
        assert self.id == other.id

        attrs = {}
        for attr in get_annotations(self.__class__, eval_str=True).keys():
            if random.random() < 0.5:
                attrs[attr] = getattr(self, attr)
            else:
                attrs[attr] = getattr(other, attr)

        return self.__class__(**attrs)

    def mutate(self, params: GenomeParams):
        for attr, attr_type in get_annotations(self.__class__, eval_str=True).items():
            if attr in self._excluded_attrs_from_mutation:
                continue

            if issubclass(attr_type, bool):
                self.mutate_bool(attr, params)
            elif issubclass(attr_type, Enum):
                self.mutate_enum(attr, params)
            elif issubclass(attr_type, float):
                self.mutate_float(attr, params)
            elif issubclass(attr_type, int):
                self.mutate_int(attr, params)
            else:
                raise NotImplementedError("Mutation attr type is not implemented.")

    def mutate_float(self, attr: str, params: GenomeParams):
        mutation_chance: float = getattr(params, f"{attr}_mutation_chance")
        replace_chance: float = getattr(params, f"{attr}_replace_chance")

        if random.random() < mutation_chance:
            min_value: int = getattr(params, f"{attr}_min_value")
            max_value: int = getattr(params, f"{attr}_max_value")

            if random.random() < replace_chance:
                init_mean: float = getattr(params, f"{attr}_init_mean")
                init_stdev: float = getattr(params, f"{attr}_init_stdev")
                value = random.gauss(init_mean, init_stdev)
            else:
                mutation_power: float = getattr(params, f"{attr}_mutation_power")
                curr_value: int = getattr(self, attr)
                change = random.gauss(0.0, mutation_power)
                value = curr_value + change

            setattr(self, attr, clamp(value, min_value, max_value))

    def mutate_int(self, attr: str, params: GenomeParams):
        mutation_chance: float = getattr(params, f"{attr}_mutation_chance")
        replace_chance: float = getattr(params, f"{attr}_replace_chance")

        if random.random() < mutation_chance:
            min_value: int = getattr(params, f"{attr}_min_value")
            max_value: int = getattr(params, f"{attr}_max_value")

            if random.random() < replace_chance:
                init_mean: float = getattr(params, f"{attr}_init_mean")
                init_stdev: float = getattr(params, f"{attr}_init_stdev")
                value = random.gauss(init_mean, init_stdev)
            else:
                mutation_power: float = getattr(params, f"{attr}_mutation_power")
                curr_value: int = getattr(self, attr)
                change = random.gauss(0.0, mutation_power)
                value = curr_value + change

            setattr(self, attr, math.ceil(clamp(value, min_value, max_value)))

    def mutate_bool(self, attr: str, params: GenomeParams):
        mutation_chance: float = getattr(params, f"{attr}_mutation_chance")

        if random.random() < mutation_chance:
            curr_value: bool = getattr(self, attr)
            setattr(self, attr, not curr_value)

    def mutate_enum(self, attr: str, params: GenomeParams):
        mutation_chance: float = getattr(params, f"{attr}_mutation_chance")

        if random.random() < mutation_chance:
            options: list[StrEnum] = getattr(params, f"{attr}_options")
            value = random.choice(options)
            setattr(self, attr, value)

    def copy(self) -> Gene:
        return copy.copy(self)

    @classmethod
    def get_default_args(cls, params: GenomeParams) -> list[Any]:
        values = []
        for attr, attr_type in get_annotations(cls, eval_str=True).items():
            if attr in cls._excluded_attrs_from_mutation:
                continue

            if issubclass(attr_type, bool):
                value = cls.get_default_bool_attr(attr, params)
            elif issubclass(attr_type, Enum):
                value = cls.get_default_enum_attr(attr, params)
            elif issubclass(attr_type, float):
                value = cls.get_default_float_attr(attr, params)
            elif issubclass(attr_type, int):
                value = cls.get_default_int_attr(attr, params)
            else:
                raise NotImplementedError("Mutation attr type is not implemented.")

            values.append(value)
        return values

    @staticmethod
    def get_default_float_attr(attr: str, params: GenomeParams) -> float:
        init_mean: float = getattr(params, f"{attr}_init_mean")
        init_stdev: float = getattr(params, f"{attr}_init_stdev")
        min_value: int = getattr(params, f"{attr}_min_value")
        max_value: int = getattr(params, f"{attr}_max_value")

        value = random.gauss(init_mean, init_stdev)
        return clamp(value, min_value, max_value)

    @staticmethod
    def get_default_int_attr(attr: str, params: GenomeParams) -> int:
        init_mean: float = getattr(params, f"{attr}_init_mean")
        init_stdev: float = getattr(params, f"{attr}_init_stdev")
        min_value: int = getattr(params, f"{attr}_min_value")
        max_value: int = getattr(params, f"{attr}_max_value")

        value = random.gauss(init_mean, init_stdev)
        return math.ceil(clamp(value, min_value, max_value))

    @staticmethod
    def get_default_bool_attr(attr: str, params: GenomeParams) -> bool:
        return getattr(params, f"{attr}_default")

    @staticmethod
    def get_default_enum_attr(attr: str, params: GenomeParams) -> Enum:
        return getattr(params, f"{attr}_default")


@dataclass(repr=False)
class Node(Gene):
    id: NodeID
    node_type: NodeType
    bias: float
    response: float
    aggregator: AggregationFuncs
    activator: ActivationFuncs

    _attrs_excluded_from_mutation = ["id", "node_type"]

    def distance(self, other: Node) -> float:
        distance = abs(self.bias - other.bias)
        distance += abs(self.response - other.response)

        if self.activator != other.activator:
            distance += 1.0

        if self.aggregator != other.aggregator:
            distance += 1.0

        return distance

    def get_evaluator(self) -> Callable[[list[float]], float]:
        aggregator = get_aggregation_func(self.aggregator)
        activator = get_activation_func(self.activator)

        if self.node_type in (NodeType.INPUT, NodeType.OUTPUT):

            def evaluate(input: list[float]) -> float:
                return aggregator(input)

        elif self.node_type == NodeType.HIDDEN:

            def evaluate(input: list[float]) -> float:
                aggregated_input = aggregator(input)
                return activator(aggregated_input * self.response + self.bias)

        else:
            msg = f"Activation of {self.node_type} is not currently implemented."
            raise NotImplementedError(msg)

        return evaluate


@dataclass(repr=False)
class Link(Gene):
    id: LinkID
    in_node: NodeID
    out_node: NodeID
    weight: float
    enabled: bool
    frozen: bool

    _attrs_excluded_from_mutation = ["id", "in_node", "out_node"]

    def distance(self, other: Link) -> float:
        distance = abs(self.weight - other.weight)

        if self.enabled != other.enabled:
            distance += 1.0

        if self.frozen != other.frozen:
            distance += 1.0

        return distance

    @property
    def simple_link(self) -> SLink:
        return (self.in_node, self.out_node)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
