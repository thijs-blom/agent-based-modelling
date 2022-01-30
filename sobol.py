# OFAT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations

# import csv
from mesa.batchrunner import BatchRunner

import numpy as np
from typing import Dict

from oneexit import OneExit

#sobol
import SALib
from SALib.sample import saltelli
from SALib.analyze import sobol

# TODO: calc_second_order = True or False ?? 
# Should we only consider first and total like what's done in slides,
# or should we also include second order like what's done in notebook?

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
    "width": 20,
    "height": 20,
    "population": 100,
    "door_size": 2,
    "time_step": 0.01,
    "relaxation_time": 1
}

model_reporters = {
    "Exit Times": lambda m: np.mean(m.exit_times)
    }

replicates = 1
max_steps = 1500
distinct_samples = 2

# Actual Sample values
# replicates = 500
# max_steps = 10000
# distinct_samples = 5

# We get all our samples here
param_values = saltelli.sample(problem, distinct_samples, calc_second_order = False)
# print(len(param_values))

# READ NOTE BELOW CODE
batch = BatchRunner(OneExit, 
                    max_steps=max_steps,
                    variable_parameters={name:[] for name in problem['names']},
                    # fixed_parameters=fixed_model_params,
                    model_reporters=model_reporters)

count = 0
data = pd.DataFrame(index=range(replicates*len(param_values)), 
                                columns=['max_speed','vision'])
data['Run'], data['Exit times'] = None, None

for i in tqdm(range(replicates)):
    for vals in tqdm(param_values):
        # Change parameters that should be integers
        vals = list(vals)

        # Transform to dict with parameter names and their values
        variable_parameters = {}
        for name, val in zip(problem['names'], vals):
            variable_parameters[name] = val

        batch.run_iteration(variable_parameters, tuple(vals), count)
        iteration_data = batch.get_model_vars_dataframe().iloc[count]
        iteration_data['Run'] = count # Don't know what causes this, but iteration number is not correctly filled

        data.iloc[count, 0:2] = vals # record paramater values
        # record runs # record measurement(s)
        data.iloc[count, 2:4] = iteration_data['Run'],iteration_data['Exit Times']

        count += 1

        # print(f'{count / (len(param_values) * (replicates)) * 100:.2f}% done')
import sys
sys.exit(0)
print(data)
# ['max_speed','vision','soc_strength','obs_strength','obs_range']

Si_exit_times = sobol.analyze(problem, data['Exit times'].values, calc_second_order=False, print_to_console=False)

print(Si_exit_times)

def plot_index(s, params, i, title=''):
    """
    Creates a plot for Sobol sensitivity analysis that shows the contributions
    of each parameter to the global sensitivity.

    Args:
        s (dict): dictionary {'S#': dict, 'S#_conf': dict} of dicts that hold
            the values for a set of parameters
        params (list): the parameters taken from s
        i (str): string that indicates what order the sensitivity is.
        title (str): title for the plot
    """

    if i == '2':
        p = len(params)
        params = list(combinations(params, 2))
        indices = s['S' + i].reshape((p ** 2))
        indices = indices[~np.isnan(indices)]
        errors = s['S' + i + '_conf'].reshape((p ** 2))
        errors = errors[~np.isnan(errors)]
    else:
        indices = s['S' + i]
        errors = s['S' + i + '_conf']
        plt.figure()

    l = len(indices)

    plt.title(title)
    plt.ylim([-0.2, len(indices) - 1 + 0.2])
    plt.yticks(range(l), params)
    plt.errorbar(indices, range(l), xerr=errors, linestyle='None', marker='o')
    plt.axvline(0, c='k')

for Si in (Si_exit_times):
    # First order
    plot_index(Si, problem['names'], '1', 'First order sensitivity')
    plt.show()

    # # Second order
    # plot_index(Si, problem['names'], '2', 'Second order sensitivity')
    # plt.show()

    # Total order
    plot_index(Si, problem['names'], 'T', 'Total order sensitivity')
    plt.show()