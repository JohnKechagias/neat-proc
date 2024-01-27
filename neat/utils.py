import random
from typing import Any, Literal

from numba import njit


@njit
def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


@njit
def randon_sign() -> Literal[-1, 1]:
    return 1 if random.random() > 0.5 else -1


@njit
def mean(values: list[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


@njit
def variance(values: list[float]) -> float:
    return sum([(v - mean(values)) ** 2 for v in values]) / len(values)


@njit
def stdev(values: list[float]) -> float:
    return variance(values) ** 2


def get_random_value(d: dict[Any, Any]) -> Any:
    return random.choice(list(d.values()))
