import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from pathlib import Path

import numpy as np

import neat


def least_squares(output: np.ndarray, correct_ouput: np.ndarray) -> float:
    return np.sum(np.square(output - correct_ouput))


def fitness_func(genome: neat.Genome, data) -> float:
    network = neat.FeedForwardNetwork.from_genome(genome)
    loss: float = 0.0

    for input, correct_output in data:
        input = np.array(input, dtype=np.float32)
        correct_output = np.array(correct_output, dtype=np.float32)

        output = np.asarray(network.activate(input))
        loss += least_squares(output, correct_output)

    return max(4.0 - loss, 0)


def main():
    config_file = Path(__file__).parent / "xor.ini"
    print(f"Config Filepath: {config_file}")
    params = neat.Parameters(config_file)
    population = neat.Population(params)

    input = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]

    genome = population.run(fitness_func, input, 300)
    print(f"The best genome is {genome.id} with fitness {genome.fitness:.3f}.")
    winner_net = neat.FeedForwardNetwork.from_genome(genome)
    input_element = input[1][0]
    output = winner_net.activate(input_element)
    print(f"Input: {input_element}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
