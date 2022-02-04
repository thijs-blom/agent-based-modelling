import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval

df = pd.DataFrame()

filename = "SA_Data/Validation_DistinctSamples20_MaxSteps10000_Repi10.csv"
df = df.append(pd.read_csv(filename, converters={'Panic level': literal_eval}))

df = df.sort_values(by=["prob_stressed"])

for i in range(1, 10):
    slice = df[i * 10 - 10: i * 10]

    mean_panic = []
    lower = []
    upper = []
    time = []
    length = np.min([len(x) for x in slice["Panic level"]])
    for j in range(length):
        mean = slice["Panic level"].apply(lambda x: x[j]).mean()
        std = slice["Panic level"].apply(lambda x: x[j]).std()
        mean_panic.append(mean)
        lower.append(mean - std)
        upper.append(mean + std)
        time.append(j / 100)

    plt.plot(time, mean_panic, c='k')
    plt.fill_between(time, lower, upper, alpha=0.7)
    plt.xlabel("Time (in s)")
    plt.ylabel("Mean Panic Index")
    plt.title("Panic level")
    plt.savefig(f"Exp_Data/Exp_plot/panic_plot[{i}].png")
