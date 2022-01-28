# OFAT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from mesa.batchrunner import BatchRunner
from exit import Exit
from wall import Wall
from dead import Dead
from human2002 import Human
from model2002 import SocialForce

import numpy as np
from typing import Dict

from oneexit import OneExit
#sobol
import SALib
from SALib.sample import saltelli
from SALib.analyze import sobol

# Define variables and bounds
parameters = {
    'names': ['max_speed','vision','soc_strength'],
    'bounds': [[4,5], [3,6], [1000,2000]]
}

replicates = 1
max_steps = 10
distinct_samples = 10

# We get all our samples here
param_values = saltelli.sample(parameters, distinct_samples)

# READ NOTE BELOW CODE
batch = BatchRunner(OneExit, 
                    max_steps=max_steps,
                    variable_parameters={name:[] for name in parameters['names']},
                    model_reporters=model_reporters)

count = 0
data = pd.DataFrame(index=range(replicates*len(param_values)), 
                                columns=['population', 'relaxation_time', 'vision'])
data['Run'], data['Exit times'] = None, None

for i in range(replicates):
    for vals in param_values: 
        # Change parameters that should be integers
        vals = list(vals)
        vals[2] = int(vals[2])
        # Transform to dict with parameter names and their values
        variable_parameters = {}
        for name, val in zip(problem['names'], vals):
            variable_parameters[name] = val

        batch.run_iteration(variable_parameters, tuple(vals), count)
        iteration_data = batch.get_model_vars_dataframe().iloc[count]
        iteration_data['Run'] = count # Don't know what causes this, but iteration number is not correctly filled
        data.iloc[count, 0:3] = vals
        data.iloc[count, 3:6] = iteration_data
        count += 1

        clear_output()
        print(f'{count / (len(param_values) * (replicates)) * 100:.2f}% done')