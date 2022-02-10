from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mesa.batchrunner import BatchRunnerMP
from socialforce.one_exit import OneExit

import numpy as np

import sys

parameters = {
    'names': ['door_size'],
    'bounds': [[0.6, 2.7]]
}


def main(pop):
    # Set the repetitions, the amount of steps, and the amount of distinct values per variable
    replicates = 10
    max_steps = 10000  # within 100 second performance
    distinct_samples = 20

    # Set up all the parameters to be entered into the model
    model_params = {
        "population": int(pop)
    }

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

        batch = BatchRunnerMP(OneExit,
                              nr_processes=8,
                              max_steps=max_steps,
                              iterations=replicates,
                              fixed_parameters=model_params,
                              variable_parameters={var: samples},
                              model_reporters=model_reporters,
                              display_progress=True)
        batch.run_all()
        file = file.append(batch.get_model_vars_dataframe())
        data[var] = batch.get_model_vars_dataframe()

    print(data)
    file.to_csv(
        Path(__file__).parent /
        f"data/Exp_Door_DistinctSamples{distinct_samples}_MaxSteps{max_steps}_Repi{replicates}_pop{pop}.csv"
    )
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

    plt.set_xlabel("Size of exit")
    plt.set_ylabel(param)


def plot_all_vars(data, param):
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

    plt.xlabel("Size of exit")
    plt.ylabel(param)
    plt.title("Exit Size Experiment")


if __name__ == "__main__":
    data = main(sys.argv[1])
    # recall the experiment values
    replicates = 10
    max_steps = 10000  # within 100 second performance
    distinct_samples = 20
    for param in ("Mean exit time", "std exit time", "Flow", "Evacuation percentage"):
        plot_all_vars(data, param)
        plt.savefig(
            Path(__file__).parent /
            f"images/Exp_Door_Outcome{param}_DistinctSamples{distinct_samples}_MaxSteps{max_steps}_Repi{replicates}_pop{sys.argv[1]}.png"
        )
        plt.show()
