from enum import StrEnum
from typing import Callable

from numba import njit


class StagnationFuncs(StrEnum):
    ABSOLUTE = "absolute"
    MIN_GROWTH = "linear_growth"


def get_stagnation_func(func: StagnationFuncs) -> Callable[[list[float]], bool]:
    stagnation_func_mapper = {
        StagnationFuncs.ABSOLUTE: absolute_stagnation_calc,
        StagnationFuncs.MIN_GROWTH: min_growth_stagnation_calc,
    }
    return stagnation_func_mapper[func]


@njit
def absolute_stagnation_calc(fitness_history: list[float]) -> bool:
    if len(fitness_history) < 2:
        raise RuntimeError("Fitness history does not have at least 2 values.")

    return fitness_history[-1] <= fitness_history[-2]


@njit
def min_growth_stagnation_calc(fitness_history: list[float], growth: float) -> bool:
    if len(fitness_history) < 2:
        raise RuntimeError("Fitness history does not have at least 2 values.")

    return fitness_history[-1] - fitness_history[-2] < growth
