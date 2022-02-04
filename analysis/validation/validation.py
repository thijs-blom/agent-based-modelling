# Validation
import matplotlib.pyplot as plt
import pandas as pd
from mesa.batchrunner import BatchRunnerMP
from model.oneexit import OneExit

import numpy as np

parameters = {
        'names': ['init_desired_speed'],
        'bounds': [[0.5, 5]],
    }

def main():

    # Set the repetitions, the amount of steps, and the amount of distinct values per variable
    replicates = 1
    max_steps = 100000
    distinct_samples = 19

    # Set up all the parameters to be entered into the model
    model_params = {
        "population": 200,
    }

    model_reporters = {
            "Exit times list": lambda m: m.exit_times,
            "Mean exit time": lambda m: np.mean(m.exit_times),
            "std exit time": lambda m: np.std(m.exit_times, ddof=1),
            "Flow / Desired Velocity": lambda m: m.flow() / m.init_desired_speed,
            "Evacuation percentage": lambda m: m.evacuation_percentage(),
            "Evacuation time": lambda m: m.evacuation_time(),
        }

    data = {}
    file = pd.DataFrame()

    for i, var in enumerate(parameters['names']):
        # Get the bounds for this variable and get <distinct_samples> samples within this space (uniform)
        samples = np.linspace(*parameters['bounds'][i], num=distinct_samples)
        
        batch = BatchRunnerMP(OneExit,
                            nr_processes=4,
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

    return data

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

    plt.xlabel("Desired Velocity (m/s)")
    plt.ylabel(r"Pedestrian Flow (m$^{-1}$s$^{-1}$) / Desired Velocity (m/s)")
    plt.title("Pedestrian Flow Compared to Desired Velocity")

if __name__ == "__main__":
    data = main()

    plot_all_vars(data, 'Flow / Desired Velocity')
    plt.savefig("validation.png")
    plt.show()