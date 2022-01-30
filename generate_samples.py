from typing import List

import numpy as np
from SALib.sample import saltelli


def generate_samples(num_samples: int = 512, second_order: bool = False):
    variable_definition = {
        'num_vars': 4,
        'names': ['max_speed', 'vision', 'soc_strength', 'obs_strength'],
        # TODO: set the bounds for global analysis
        'bounds': [[3, 5], [1, 5], [1000, 3000], [2000, 5000]]
    }

    return saltelli.sample(variable_definition, num_samples, calc_second_order=second_order)


def main(names: List[str]):
    # Calculate the samples
    samples = generate_samples(num_samples=512, second_order=False)

    # Divide the samples into partitions
    num_partitions = len(names)
    partitions = np.array_split(np.array(samples), num_partitions)

    for name, partition in zip(names, partitions):
        np.save(f"samples/samples_{name}", partition)


if __name__ == "__main__":
    main(['Thijs', 'Rina', 'Liza', 'Mercylyn', 'Tamara'])
