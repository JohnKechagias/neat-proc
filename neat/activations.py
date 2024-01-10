import math
from enum import StrEnum
from typing import Callable

from numba import njit


class ActivationFuncs(StrEnum):
    SIGMOID = "sigmoid"
    RELU = "relu"
    TANH = "tanh"
    LINEAR = "linear"


def get_activation_func(func: ActivationFuncs) -> Callable[[float], float]:
    activation_func_mapper = {
        ActivationFuncs.SIGMOID: sigmoid,
        ActivationFuncs.RELU: relu,
        ActivationFuncs.TANH: tanh,
    }
    return activation_func_mapper[func]


@njit
def relu(value: float) -> float:
    return max(value, 0.0)


@njit
def sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, 5.0 * value))
    return 1.0 / (1.0 + math.exp(-value))


@njit
def tanh(value: float) -> float:
    value = max(-60.0, min(60.0, 2.5 * value))
    return math.tanh(value)


@njit
def linear(value: float) -> float:
    return value
