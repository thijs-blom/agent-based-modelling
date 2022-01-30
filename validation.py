# OFAT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from mesa.batchrunner import BatchRunner
from oneexit import OneExit

import numpy as np
from typing import Dict

parameters = {
    'names': ['init_desired_speed'],
    'bounds': [[0.5, 5]]
}

# Set the repetitions, the amount of steps, and the amount of distinct values per variable
replicates = 1
max_steps = 10000
distinct_samples = 10

# Set up all the parameters to be entered into the model
model_params = {
    "width": 15,
    "height": 15,
    "population": 200,
    "vision": 1,
    "max_speed": 5,
    "timestep": 0.01,
    "prob_nearest": 1,
}

model_reporters = {
    "Flow / Desired Velocity": lambda m: m.flow / m.init_desired_speed,
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
                        fixed_parameters=model_params,
                        variable_parameters={var: samples},
                        model_reporters= model_reporters,
                        display_progress=True)
    batch.run_all()
    file = file.append(batch.get_model_vars_dataframe())
    data[var] = batch.get_model_vars_dataframe()

print(data)
file.to_csv(f"SA_Data/Validation_DistinctSamples{distinct_samples}_MaxSteps{max_steps}_Repi{replicates}.csv")

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

    plt.plot(x, y, c='k')
    plt.fill_between(x, y - err, y + err)

    plt.set_xlabel("Desired Velocity")
    plt.set_ylabel(param)

def plot_all_vars(df, param):
    """
    Plots the parameters passed vs each of the output variables.

    Args:
        df: dataframe that holds all data
        param: the parameter to be plotted
    """
    
    var = parameters['names'][0]
    df = data[var]

    x = df.groupby(var).mean().reset_index()[var]
    y = df.groupby(var).mean()[param]

    replicates = df.groupby(var)[param].count()
    err = (1.96 * df.groupby(var)[param].std()) / np.sqrt(replicates)

    plt.plot(x, y, c='k')
    plt.fill_between(x, y - err, y + err)

    plt.xlabel("Desired Velocity")
    plt.ylabel(param)
    plt.title("Validation of our model")


plot_all_vars(data, 'Flow / Desired Velocity')
plt.show()
