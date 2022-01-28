# OFAT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from mesa.batchrunner import BatchRunner
# from exit import Exit
# from wall import Wall
# from dead import Dead
# from human2002 import Human
# from model2002 import SocialForce

import numpy as np
from typing import Dict

from oneexit import OneExit
from server2002 import width, height

#sobol
import SALib
from SALib.sample import saltelli
from SALib.analyze import sobol

# Define variables and bounds
# problem = {
#     'num_vars' : 5,
#     'names': ['max_speed','vision','soc_strength','obs_strength','obs_range'],
#     'bounds': [[4,5], [3,6], [1000,2000],[2000,5000],[0.06,0.10]]
# }

problem = {
    'num_vars' : 2,
    'names': ['max_speed','vision'],
    'bounds': [[4,5], [3,6]]
}

fixed_model_params = {
    "width": width,
    "height": height,
    "population": 100,
    "door_size": 2,
    "time_step": 0.01,
    "relaxation_time": 1
}

model_reporters = {
    "Exit Times": lambda m: np.mean(m.exit_times),
    "Evacuation Time": lambda m: m.evacuation_time,
    }

replicates = 10
max_steps = 10
distinct_samples = 2
# distinct_samples = 500 -> actual sample size


# We get all our samples here
param_values = saltelli.sample(problem, distinct_samples, calc_second_order = False )
print(len(param_values))

# READ NOTE BELOW CODE
batch = BatchRunner(OneExit, 
                    max_steps=max_steps,
                    variable_parameters={name:[] for name in problem['names']},
                    fixed_parameters=fixed_model_params,
                    model_reporters=model_reporters)

count = 0
data = pd.DataFrame(index=range(replicates*len(param_values)), 
                                columns=['max_speed','vision'])
data['Run'], data['Exit times'], data['Evacuation Time'] = None, None, None

for i in range(replicates):
    for vals in param_values: 
        # Change parameters that should be integers
        vals = list(vals)
        # vals[2] = int(vals[2])
        # Transform to dict with parameter names and their values
        variable_parameters = {}
        for name, val in zip(problem['names'], vals):
            variable_parameters[name] = val

        batch.run_iteration(variable_parameters, tuple(vals), count)
        iteration_data = batch.get_model_vars_dataframe().iloc[count]
        iteration_data['Run'] = count # Don't know what causes this, but iteration number is not correctly filled
        # data.iloc[count, 0:5] = vals
        # data.iloc[count, 5:8] = iteration_data
        data.iloc[count, 0:2] = vals
        data.iloc[count, 2:5] = iteration_data
        count += 1

        print(f'{count / (len(param_values) * (replicates)) * 100:.2f}% done')

# ['max_speed','vision','soc_strength','obs_strength','obs_range']

Si_max_speed = sobol.analyze(problem, data['max_speed'].values, print_to_console=True)
Si_vision = sobol.analyze(problem, data['vision'].values, print_to_console=True)
# Si_soc_strength = sobol.analyze(problem, data['soc_strength'].values, print_to_console=True)
# Si_obs_strength = sobol.analyze(problem, data['obs_strength'].values, print_to_console=True)
# Si_obs_range = sobol.analyze(problem, data['obs_range'].values, print_to_console=True)
print(Si_max_speed)
print(Si_vision)