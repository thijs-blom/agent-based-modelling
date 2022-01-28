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
# from base_human import Human
from oneexit import OneExit

import numpy as np
from typing import Dict

# from server2002 import width, height

#sobol
# import SALib
# from SALib.sample import saltelli
# from SALib.analyze import sobol

# Define variables and bounds
parameters = {
    'names': ['population', 'relaxation_time', 'door_size'],
    'bounds': [[10, 200], [0.5, 0.1], [0.6, 2.4]]
}

# Set the repetitions, the amount of steps, and the amount of distinct values per variable
replicates = 2
max_steps = 10
distinct_samples = 4

# Set up all the parameters to be entered into the model
model_params = {
    "width": 20,
    "height": 20,
    "vision": 1,
    "max_speed": 5,
    "timestep": 0.01
}

model_reporters = {
    #"Number of Humans in Environment": lambda m: m.schedule.get_agent_count(),
    # "Number of Casualties": lambda m: len(self.obstacles) - self.init_amount_obstacles,
    # "Average Energy": lambda m: self.count_energy(m) / self.population,
    #"Average Speed": lambda m: m.count_speed() / m.schedule.get_agent_count() if m.schedule.get_agent_count() > 0 else 0
    "Exit Times": lambda m: np.mean(m.exit_times),
    "Evacuation Time": lambda m: m.evacuation_time,
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

# file.to_csv(f"OFAT_DistinctSamples{distinct_samples}_MaxSteps{max_steps}_Repi{replicates}.csv")

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

for params in ('Exit Times', 'Evacuation Time'):
    plot_all_vars(data, params)
    plt.show()

file.to_csv(f"SA_Data\OFAT_DistinctSamples{distinct_samples}_MaxSteps{max_steps}_Repi{replicates}.csv")
