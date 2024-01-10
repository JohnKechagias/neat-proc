from enum import StrEnum
from typing import Callable

import numpy as np
from numba import njit


class FitnessCriterionFuncs(StrEnum):
    MAX = "max"


class LossFuncs(StrEnum):
    LEAST_SQUARES = "least_squares"


def get_loss_func(func: LossFuncs) -> Callable[[np.ndarray, np.ndarray], float]:
    loss_func_mapper = {
        LossFuncs.LEAST_SQUARES: least_squares,
    }

    return loss_func_mapper[func]


@njit
def least_squares(output: np.ndarray, correct_ouput: np.ndarray) -> float:
    return np.sum(np.square(output - correct_ouput))
