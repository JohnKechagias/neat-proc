from pathlib import Path

DATAFILES = Path(__file__).parent / "datafiles"

CONFIG_FILEPATH = DATAFILES / "example_config.ini"

correct_config_representation = """[Parameters]

[NEATParameters]
population = 150
reset_on_extinction = True

[GenomeParameters]
number_of_inputs = 2
number_of_outputs = 2
number_of_hidden_nodes = 0
feed_forward = True
connection_scheme = full
activation_default = sigmoid
activation_options = sigmoid
activation_mutation_chance = 0.0
aggregation_default = sum
aggregation_options = sum
aggregation_mutation_chance = 0.0
link_mutation_chance = 1.0
link_addition_chance = 0.2
link_deletion_chance = 0.2
link_toggle_chance = 0.2
node_mutation_chance = 1.0
node_addition_chance = 0.2
node_deletion_chance = 0.2
bias_init_mean = 0.0
bias_init_stdev = 1.0
bias_min_value = -30.0
bias_max_value = 30.0
bias_mutation_chance = 0.7
bias_replace_chance = 0.1
bias_mutation_power = 0.5
response_init_mean = 1.0
response_init_stdev = 0.0
response_min_value = -30.0
response_max_value = 30.0
response_mutation_chance = 0.0
response_replace_chance = 0.0
response_mutation_power = 0.0
weight_init_mean = 0.0
weight_init_stdev = 1.0
weight_min_value = -30.0
weight_max_value = 30.0
weight_mutation_chance = 0.8
weight_severe_mutation_chance = 0.1
weight_replace_chance = 0.1
weight_mutation_power = 0.5

[SpeciationParameters]
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.6
compatibility_threshold = 3.0
species_fitness_func = max
species_elitism = 100
max_stagnation = 5
survival_rate = 0.2
elitism = 5

[EvaluationParameters]
fitness_threshold = 3.9
fitness_criterion = max
loss_function = least_squares

[ReproductionParameters]
crossover_rate = 0.4"""
