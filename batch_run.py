import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
from mesa.batchrunner import BatchRunnerMP

from oneexit import OneExit


def batch_run(samples: np.ndarray,
              max_steps: int,
              fixed_model_params: Dict,
              model_reporters: Dict,
              processes: int) -> pd.DataFrame:
    # Define the variable parameters used in the model (max_speed, vision, soc_strength, obs_strength)
    variable_params = {"variable_parameters": samples}

    # These parameters are passed using variable_params, and _must_ themselves be set to None.
    fixed_params = fixed_model_params.copy()
    fixed_params["max_speed"] = None
    fixed_params["vision"] = None
    fixed_params["soc_strength"] = None
    fixed_params["obs_strength"] = None

    # Define the batch runner using multiple processes
    batch = BatchRunnerMP(OneExit,
                          nr_processes=processes,
                          max_steps=max_steps,
                          variable_parameters=variable_params,
                          fixed_parameters=fixed_params,
                          model_reporters=model_reporters)

    # Start the parallel processing of the samples
    batch.run_all()

    # Return a Pandas dataframe with the parameters, and the specified model reporters
    return batch.get_model_vars_dataframe()


def main(input_file: str,
         output_file: str,
         processes: int = 4):

    # Load the generated samples
    samples = np.load(f"samples/{input_file}")

    # The maximum number of ticks the simulation will run for
    # TODO: define the number of steps
    max_steps = 1500

    # Define (non-default) fixed model parameters
    fixed_model_params = {
        "door_size": 2,
        "time_step": 0.01,
        "relaxation_time": 1
    }

    # Define the statistics we want to collect after a simulation
    model_reporters = {
        "Exit Times": lambda m: np.mean(m.exit_times)
    }

    # Run the actual samples
    df = batch_run(samples, max_steps, fixed_model_params, model_reporters, processes)

    # Write the results
    df.to_csv(output_file)


if __name__ == "__main__":
    # Check if a filename has been passed
    if len(sys.argv) != 3:
        raise ValueError("Specify the filename as the first argument to collect " +
                         "the right samples (see the samples directory)")

    filename = sys.argv[1]
    if not os.path.exists(f"samples/{filename}"):
        raise ValueError("The specified file does not exist. Make sure you have pulled the samples from git " +
                         "and passed the correct filename.")

    num_processes = sys.argv[2]
    if not num_processes.isdigit():
        raise ValueError("Number of processes must be an integer")
    num_processes = int(num_processes)

    main(input_file=f"samples/{filename}", output_file=f"data/{filename}", processes=num_processes)
