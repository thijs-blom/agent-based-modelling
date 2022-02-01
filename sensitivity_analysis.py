# OFAT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from mesa.batchrunner import BatchRunner
from oneexit import OneExit

import numpy as np
from typing import Dict

#sobol
# import SALib
# from SALib.sample import saltelli
# from SALib.analyze import sobol

# Define variables and bounds
parameters = {
    'names': ['population', 'relaxation_time', 'door_size'],
    'bounds': [[10, 333], [0.06, 0.81], [0.6, 2.7]]
}

# Set the repetitions, the amount of steps, and the amount of distinct values per variable
replicates = 10
max_steps = 10000
distinct_samples = 20

# Set up all the parameters to be entered into the model
# model_params = {
#     "width": 15,
#     "height": 15,
#     "population": 100,
#     "vision": 1,
#     "max_speed": 5,
#     "timestep": 0.01,
#     "prob_nearest": 1,
# }

model_reporters = {
        "Mean exit time": lambda m: np.mean(m.exit_times),
        "std exit time": lambda m: np.std(m.exit_times, ddof=1),
        "Flow": lambda m: m.flow(),
        "Evacuation percentage": lambda m: m.evacuation_percentage(),
        "Evacuation time": lambda m: m.evacuation_time(),
    }

data = {}
file = pd.DataFrame()

for i, var in enumerate(parameters['names']):
    # Get the bounds for this variable and get <distinct_samples> samples within this space (uniform)
    samples = np.linspace(*parameters['bounds'][i], num=distinct_samples)
    
    # Keep in mind that wolf_gain_from_food should be integers. You will have to change
    # your code to acommodate for this or sample in such a way that you only get integers.
    if var == 'population':
        samples = np.linspace(*parameters['bounds'][i], num=distinct_samples, dtype=int)
    
    batch = BatchRunner(OneExit,
                        max_steps=max_steps,
                        iterations=replicates,
                        variable_parameters={var: samples},
                        model_reporters= model_reporters,
                        display_progress=True)
    batch.run_all()
    file = file.append(batch.get_model_vars_dataframe())
    data[var] = batch.get_model_vars_dataframe()

file.to_csv(f"SA_Data/OFAT_DistinctSamples{distinct_samples}_MaxSteps{max_steps}_Repi{replicates}.csv")

# file.to_csv(f"OFAT_DistinctSamples{distinct_samples}_MaxSteps{max_steps}_Repi{replicates}.csv")

# put all the sa analysis to jupyter file later for ploting 
def plot_param_var_conf(ax, df, var, param, i):
    """
    Helper function for plot_all_vars. Plots the individual parameter vs
    variables passed.

    Args:
        ax: the axis to plot to
        df: dataframe that holds the data to be plotted
        var: variables to be taken from the dataframe
        param: which output variable to plot
    """
    x = df.groupby(var).mean().reset_index()[var]
    y = df.groupby(var).mean()[param]

    replicates = df.groupby(var)[param].count()
    err = (1.96 * df.groupby(var)[param].std()) / np.sqrt(replicates)

    ax.plot(x, y, c='k')
    ax.fill_between(x, y - err, y + err)

    ax.set_xlabel(var)
    ax.set_ylabel(param)

def plot_all_vars(df, params):
    """
    Plots the parameters passed vs each of the output variables.

    Args:
        df: dataframe that holds all data
        param: the parameter to be plotted
    """

    f, axs = plt.subplots(3, figsize=(7, 10))
    
    for i, var in enumerate(parameters['names']):
        plot_param_var_conf(axs[i], data[var], var, params, i)

for params in ("Mean exit time","std exit time", "Flow", "Evacuation percentage", "Evacuation time"):
    plot_all_vars(data, params)
    plt.savefig(f'SA_Data/OFAT_ParamName{params}_DistinctSamples{distinct_samples}_MaxSteps{max_steps}_Repi{replicates}.jpg')
