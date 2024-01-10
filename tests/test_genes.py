from neat.genomes.genome import Genome
from neat.innovation import InnovationRecord
from neat.parameters import Parameters
from tests.data import CONFIG_FILEPATH


params = Parameters(CONFIG_FILEPATH)
def test_genome_initialization():
    innov_record = InnovationRecord(params.genome.inputs, params.genome.outputs)
    Genome.initialize_configuration(params.genome)
    genome = Genome(0, innov_record)

    assert list(genome.links.keys()) == [0, 1, 2, 3]
    assert list(genome.nodes.keys()) == list(range(params.genome.inputs + params.genome.outputs))
