import matplotlib.pyplot as plt
import pandas as pd
from mesa.batchrunner import BatchRunnerMP
from socialforce.one_exit import OneExit

import numpy as np

parameters = {
    'names': 'strategy_weights',
    'bounds': [0.1, 1.0]
}


def main():
    # Set the repetitions, the amount of steps, and the amount of distinct values per variable
    replicates = 5

    # Run for 100 seconds maximum
    max_steps = 100
    distinct_samples = 10

    # Set up all the parameters to be entered into the model
    model_params = {
        "vision": 2,
        "population": 200,
        "strategies": ['nearest exit', 'follow the leader'],
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

    # Get the bounds for this variable and get <distinct_samples> samples within this space (uniform)
    prob = np.linspace(*parameters['bounds'], num=distinct_samples)
    prob = [round(x, 1) for x in prob]
    samples = [[x, 1 - x] for x in prob]
    print(samples)
    batch = BatchRunnerMP(OneExit,
                            nr_processes=4,
                            max_steps=max_steps,
                            iterations=replicates,
                            fixed_parameters=model_params,
                            variable_parameters={parameters['names']: samples},
                            model_reporters=model_reporters,
                            display_progress=True)
    batch.run_all()
    file = file.append(batch.get_model_vars_dataframe())
    data[parameters['names']] = batch.get_model_vars_dataframe()

    print(data)
    file.to_csv(
        f"data/Exp_Prob_DistinctSamples{distinct_samples}_MaxSteps{max_steps}_Repi{replicates}_Vision2_pop200.csv")
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

    plt.set_xlabel("Probability of 'Nearest Exit'")
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

    plt.xlabel("Probability of 'Nearest Exit'")
    plt.ylabel(param)
    plt.title("Strategy Experiment")


if __name__ == "__main__":
    data = main()
    # recall the experiment values
    replicates = 10
    max_steps = 10000  # within 100 second performance
    distinct_samples = 11
    for param in ("Mean exit time", "std exit time", "Flow", "Evacuation percentage"):
        plot_all_vars(data, param)
        plt.savefig(
            f"images/Exp_Prob_Outcome{param}_DistinctSamples{distinct_samples}_MaxSteps{max_steps}_Repi{replicates}_Vision2_pop200.png")
        plt.show()
