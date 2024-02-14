import inspect
import multiprocessing
import os
import sys
from functools import partial

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from pathlib import Path

import numpy as np

import neat
from neat.evaluators import ParallelEvaluator


def least_squares(output: np.ndarray, correct_ouput: np.ndarray) -> float:
    return np.sum(np.square(output - correct_ouput))


def fitness_func(data, genome: neat.Genome, _) -> float:
    network = neat.FeedForwardNetwork.from_genome(genome)
    loss: float = 0.0

    for input, correct_output in data:
        input = np.array(input, dtype=np.float32)
        correct_output = np.array(correct_output, dtype=np.float32)

        output = np.asarray(network.activate(input))
        loss += least_squares(output, correct_output)

    return max(4.0 - loss, 0.0)


def main():
    config_file = Path(__file__).parent / "xor.ini"
    print(f"Config Filepath: {config_file}")
    params = neat.Parameters(config_file)
    population = neat.Population(params)

    inputs = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]

    eval_func = partial(fitness_func, inputs)
    evaluator = ParallelEvaluator(multiprocessing.cpu_count(), eval_func)

    genome = population.run(evaluator.evaluate, 200)
    print(f"The best genome is {genome.id} with fitness {genome.fitness:.3f}.")
    winner_net = neat.FeedForwardNetwork.from_genome(genome)

    for input, correct_output in inputs:
        output = winner_net.activate(np.asarray(input))
        print(f"{input} -> {output} instead of {correct_output}")


if __name__ == "__main__":
    main()
