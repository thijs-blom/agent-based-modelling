from typing import List

import numpy as np
from SALib.sample import saltelli


def generate_samples(num_samples: int = 512, second_order: bool = False) -> np.ndarray:
    """Generate samples for global sensitivity analysis

    Args:
        num_samples: The number of samples to generate. Note that this generates num_samples * (D + 2)
                     combinations of parameters, where D is the number of variables.
        second_order: Boolean indicating whether second-order sensitivities should be calculated.

    Returns:
        A list of samples of length num_samples * 6, where each sample is a list of four elements
        for max_speed, vision, soc_strength and obs_strength respectively.
    """
    variable_definition = {
        'num_vars': 4,
        'names': ['max_speed', 'vision', 'soc_strength', 'obs_strength'],
        'bounds': [[3, 5], [1, 5], [1000, 3000], [2000, 5000]]
    }

    return saltelli.sample(variable_definition, num_samples, calc_second_order=second_order)


def main(names: List[str], repetitions: int):
    """Main function to generate samples and store for global sensitivity analysis
    Args:
        names: The names of the people who will run the simulations for the specific samples
        repetitions: The number of times a sample must be run
    """
    # Calculate the samples
    samples = generate_samples(num_samples=512, second_order=False)

    # Divide the samples into partitions
    num_partitions = len(names)
    partitions = np.array_split(np.array(samples), num_partitions)

    for name, partition in zip(names, partitions):
        for i in range(repetitions):
            np.save(f"samples/samples_{name}_{i + 1}", partition)


if __name__ == "__main__":
    main(['Thijs', 'Rina', 'Liza', 'Mercylyn', 'Tamara'], 5)
