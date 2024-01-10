from neat.parameters import Parameters
from tests.data import CONFIG_FILEPATH, correct_config_representation


def test_read_parameters():
    params = Parameters(CONFIG_FILEPATH)
    assert str(params) == correct_config_representation
