import argparse
import os
from typing import Dict

import numpy as np
import pandas as pd
from mesa.batchrunner import BatchRunnerMP

from one_exit_sample import OneExitSample
from sample import Sample


def batch_run(samples: np.ndarray,
              max_steps: int,
              model_reporters: Dict,
              processes: int) -> pd.DataFrame:
    # Define the variable parameters used in the model (max_speed, vision, soc_strength, obs_strength)
    sample_list = [Sample(*values) for values in samples]
    variable_params = {"sample": sample_list}

    # Define the batch runner using multiple processes
    batch = BatchRunnerMP(OneExitSample,
                          nr_processes=processes,
                          max_steps=max_steps,
                          variable_parameters=variable_params,
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

    # TODO: check if this is necessary
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
    df = batch_run(samples, max_steps, model_reporters, processes)

    # Write the results
    df.to_csv(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a batch of simulations for global sensitivity analysis.')

    parser.add_argument('filename', type=str, help="Name of the file containing the samples")
    parser.add_argument('nr_processes', type=int, help="Number of processes to use for computation")

    args = parser.parse_args()

    if not os.path.exists(f"samples/{args.filename}"):
        raise ValueError("The specified file does not exist. Make sure you have pulled the samples from git " +
                         "and passed the correct filename.")

    # Replace the extension with csv for the output file
    filename_out = args.filename.split('.')[0] + '.csv'

    main(input_file=f"samples/{args.filename}", output_file=f"data/{filename_out}", processes=args.nr_processes)
