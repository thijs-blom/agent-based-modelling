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
# from base_human import Human
from model2002 import SocialForce

import numpy as np
from typing import Dict

from server2002 import width, height, init_obstacles, exit2

#sobol
<<<<<<< HEAD
# import SALib
# from SALib.sample import saltelli
# from SALib.analyze import sobol
=======
import SALib
from SALib.sample import saltelli
from SALib.analyze import sobol
>>>>>>> b8215612997d8151ad085a583bd093124ffb0cc9


# Define variables and bounds
parameters = {
    'names': ['population', 'relaxation_time', 'vision'],
    'bounds': [[10, 1000], [0.5, 0.1], [1, 10]]
}

parameters = {
    'names': ['population'],
    'bounds': [[10, 1000]]
}


# Set the repetitions, the amount of steps, and the amount of distinct values per variable
# replicates = 30
# max_steps = 100
distinct_samples = 2

# Set up all the parameters to be entered into the model
model_params = {
    "width": width,
    "height": height,
    "obstacles": init_obstacles,
    "exits": [exit2]
}

model_reporters = {
    #"Number of Humans in Environment": lambda m: m.schedule.get_agent_count(),
    # "Number of Casualties": lambda m: len(self.obstacles) - self.init_amount_obstacles,
    # "Average Energy": lambda m: self.count_energy(m) / self.population,
    #"Average Speed": lambda m: m.count_speed() / m.schedule.get_agent_count() if m.schedule.get_agent_count() > 0 else 0
    "Exit times": lambda m: m.exit_times
    }

data = {}

for i, var in enumerate(parameters['names']):
    # Get the bounds for this variable and get <distinct_samples> samples within this space (uniform)
    samples = np.linspace(*parameters['bounds'][i], num=distinct_samples)
    
    # Keep in mind that wolf_gain_from_food should be integers. You will have to change
    # your code to acommodate for this or sample in such a way that you only get integers.
    if var == 'population':
        samples = np.linspace(*parameters['bounds'][i], num=distinct_samples, dtype=int)
    
    batch = BatchRunner(SocialForce,
                        max_steps=10,
                        iterations=1,
                        fixed_parameters=model_params,
                        variable_parameters={var: samples},
                        model_reporters= model_reporters,
                        display_progress=True)
    batch.run_all()

    data[var] = batch.get_model_vars_dataframe()
print(data)
# file = open("test.csv", "w")
# file.write(data)
# file.close()

