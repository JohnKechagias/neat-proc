from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from typing import Callable, Optional

from .genomes import Genome


class ParallelEvaluator:
    def __init__(
        self,
        num_workers: int,
        eval_function: Callable[[Genome, list[Genome]], float],
        timeout: Optional[float] = None,
    ):
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(num_workers)

    def __del__(self):
        self.pool.close()
        self.pool.join()

    def evaluate(self, genomes: list[Genome], opponents: list[Genome]):
        jobs: list[AsyncResult] = []
        for genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, opponents)))

        # assign the fitness back to each genome
        for job, genome in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)
