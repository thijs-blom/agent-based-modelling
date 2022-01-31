import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
from mesa.batchrunner import BatchRunnerMP

from oneexit import OneExit
from sample import Sample


def batch_run(samples: np.ndarray,
              max_steps: int,
              fixed_model_params: Dict,
              model_reporters: Dict,
              processes: int) -> pd.DataFrame:
    # Define the variable parameters used in the model (max_speed, vision, soc_strength, obs_strength)
    sample_list = [Sample(*values) for values in samples]
    variable_params = {"sample": sample_list}

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

    # Get a Pandas dataframe with the parameters, and the specified model reporters
    df = batch.get_model_vars_dataframe()

    # Set the parameters properly, instead of it being a Sample object
    df["max_speed"] = df["sample"].apply(lambda s: s.max_speed)
    df["vision"] = df["sample"].apply(lambda s: s.vision)
    df["soc_strength"] = df["sample"].apply(lambda s: s.soc_strength)
    df["obs_strength"] = df["sample"].apply(lambda s: s.obs_strength)
    df = df.drop(columns=["sample"])

    # Reorder the columns to give the same dataframe as if the sample parameters
    # were directly passed as variable parameters. May not be necessary, but might
    # be handy for consistency.
    cols = df.columns.tolist()
    cols = cols[-4:] + cols[:-4]
    df = df[cols]

    return df


def main(input_file: str,
         output_file: str,
         processes: int = 4):

    # Load the generated samples
    samples = np.load(input_file)

    # The maximum number of ticks the simulation will run for
    # TODO: define the number of steps
    max_steps = 10000

    # Define the statistics we want to collect after a simulation
    model_reporters = {
        "Mean exit time": lambda m: np.mean(m.exit_times),
        "std exit time": lambda m: np.std(m.exit_times, ddof=1),
        "Flow": lambda m: m.flow(),
        "Evacuation percentage": lambda m: m.evacuation_percentage(),
        "Evacuation time": lambda m: m.evacuation_time(),
    }

    # Run the actual samples
    df = batch_run(samples, max_steps, {}, model_reporters, processes)

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
    # Replace the extension with csv for the output file
    filename_out = filename.split('.')[0] + '.csv'

    num_processes = sys.argv[2]
    if not num_processes.isdigit():
        raise ValueError("Number of processes must be an integer")
    num_processes = int(num_processes)

    main(input_file=f"samples/{filename}", output_file=f"data/{filename_out}", processes=num_processes)
