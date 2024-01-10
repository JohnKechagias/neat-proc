import collections
import re
import sys
from configparser import ConfigParser
from dataclasses import dataclass
from inspect import get_annotations
from pathlib import Path
from types import GenericAlias
from typing import Annotated, Any, Optional, Type

from neat.activations import ActivationFuncs
from neat.aggregations import AggregationFuncs
from neat.fitness import FitnessCriterionFuncs, LossFuncs
from neat.genomes.connection_schemes import ConnectionSchemes


@dataclass
class ValueRange:
    lo: float
    hi: float


class CaseInsensitiveDict(collections.UserDict):
    """Ordered case insensitive mutable mapping class."""

    def __setitem__(self, key: str, value: Any):
        key = self.convert_to_snake_case(key)
        super().__setitem__(key, value)

    def __getitem__(self, key: str) -> Any:
        key = self.convert_to_snake_case(key)
        return super().__getitem__(key)

    @staticmethod
    def convert_to_snake_case(key: str) -> str:
        """Converts the key to lower case and replaces camel case and
        spaces with snake case.

        >>> convert_to_snake_case("ThisIs ASentence")
        "this_is_a_sentence"
        """
        key_words = re.sub(r"([A-Z][a-z])", r" \1", key).split()
        key_words = [word.lower() for word in key_words]
        return "_".join(key_words)


class ConfigSection:
    @classmethod
    def populate(cls, params: ConfigParser, section: str):
        config = cls()

        for attr, attr_type in get_annotations(cls).items():
            input_value = params.get(section, attr, fallback=None)
            if input_value is None:
                raise ValueError(f"Parameter '{attr}' is not defined in config file.")

            if enum_type := cls.get_types_from_list_annotation(attr_type):
                values = input_value.split(",")
                values = [enum_type(value) for value in values]
                setattr(config, attr, values)
                continue

            setattr(config, attr, attr_type(input_value))

        return config

    def __repr__(self) -> str:
        repr = [f"[{self.__class__.__name__}]"]

        for attr in get_annotations(self.__class__).keys():
            value = getattr(self, attr)

            if isinstance(value, list):
                value = ",".join(value)

            repr.append(f"{attr} = {value}")

        return "\n".join(repr)

    @classmethod
    def get_types_from_list_annotation(cls, value: Any) -> Optional[Type]:
        if not isinstance(value, GenericAlias):
            return

        value = str(value)
        if not (r"list[" == value[:5] and value[-1] == "]"):
            return

        # The type name that is surrounded by list[].
        type_str = value[5:-1].rsplit(".", maxsplit=1)[-1]
        attr_type = getattr(sys.modules[__name__], type_str)
        return attr_type


class NEATParams(ConfigSection):
    population: int
    reset_on_extinction: bool


class GenomeParams(ConfigSection):
    inputs: int
    outputs: int
    hidden_nodes: int

    feed_forward: bool
    connection_scheme: ConnectionSchemes
    alternative_structural_mutations: bool

    activation_default: ActivationFuncs
    activation_options: list[ActivationFuncs]
    activation_mutation_chance: Annotated[float, ValueRange(0.0, 1.0)]

    aggregation_default: AggregationFuncs
    aggregation_options: list[AggregationFuncs]
    aggregation_mutation_chance: Annotated[float, ValueRange(0.0, 1.0)]

    link_mutation_chance: Annotated[float, ValueRange(0.0, 1.0)]
    link_addition_chance: Annotated[float, ValueRange(0.0, 1.0)]
    link_deletion_chance: Annotated[float, ValueRange(0.0, 1.0)]
    link_toggle_chance: Annotated[float, ValueRange(0.0, 1.0)]

    node_mutation_chance: Annotated[float, ValueRange(0.0, 1.0)]
    node_addition_chance: Annotated[float, ValueRange(0.0, 1.0)]
    node_deletion_chance: Annotated[float, ValueRange(0.0, 1.0)]

    bias_init_mean: float
    bias_init_stdev: float
    bias_min_value: float
    bias_max_value: float
    bias_mutation_chance: Annotated[float, ValueRange(0.0, 1.0)]
    bias_replace_chance: Annotated[float, ValueRange(0.0, 1.0)]
    bias_mutation_power: float

    response_init_mean: float
    response_init_stdev: float
    response_min_value: float
    response_max_value: float
    response_mutation_chance: Annotated[float, ValueRange(0.0, 1.0)]
    response_replace_chance: Annotated[float, ValueRange(0.0, 1.0)]
    response_mutation_power: float

    weight_init_mean: float
    weight_init_stdev: float
    weight_min_value: float
    weight_max_value: float
    weight_mutation_chance: Annotated[float, ValueRange(0.0, 1.0)]
    weight_severe_mutation_chance: Annotated[float, ValueRange(0.0, 1.0)]
    weight_replace_chance: Annotated[float, ValueRange(0.0, 1.0)]
    weight_mutation_power: float


class SpeciationParams(ConfigSection):
    compatibility_disjoint_coefficient: float
    compatibility_weight_coefficient: float
    compatibility_threshold: float

    species_fitness_func: str
    species_elitism: int
    max_stagnation: int

    survival_rate: Annotated[float, ValueRange(0.0, 1.0)]
    elitism: int  # Can also be interpreted as the minimum size of a species.


class EvaluationParams(ConfigSection):
    fitness_threshold: float
    fitness_criterion: FitnessCriterionFuncs
    loss_function: LossFuncs


class ReproductionParams(ConfigSection):
    crossover_rate: Annotated[float, ValueRange(0.0, 1.0)]
    inter_species_crossover_rate: Annotated[float, ValueRange(0.0, 1.0)]
    max_stagnation: int
    survival_rate: Annotated[float, ValueRange(0.0, 1.0)]
    elitism: int  # Can also be interpreted as the minimum size of a species.
    elitism_threshold: int
    population: int


class Parameters:
    def __init__(self, config_file: Path):
        params = ConfigParser(dict_type=CaseInsensitiveDict)
        params.read(config_file)

        for section, attr_type in get_annotations(self.__class__).items():
            attr_type: ConfigSection
            if not section in params.sections():
                raise ValueError(f"Section '{section}' not present in config.")

            value = attr_type.populate(params, section)
            setattr(self, section, value)

    def __repr__(self) -> str:
        repr = []

        for attr in get_annotations(self.__class__).keys():
            value = getattr(self, attr)
            repr.append(f"\n{str(value)}")

        return "\n".join(repr)

    neat: NEATParams
    genome: GenomeParams
    speciation: SpeciationParams
    evaluation: EvaluationParams
    reproduction: ReproductionParams
