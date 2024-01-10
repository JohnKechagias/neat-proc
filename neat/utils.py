import random
from typing import Any, Literal


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def randon_sign() -> Literal[-1, 1]:
    return 1 if random.random() > 0.5 else -1


def mean(values: list[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def variance(values: list[float]) -> float:
    values = list(values)
    return sum((v - mean(values)) ** 2 for v in values) / len(values)


def stdev(values: list[float]) -> float:
    return variance(values) ** 2


def get_random_value(d: dict[Any, Any]) -> Any:
    return random.choice(list(d.values()))
