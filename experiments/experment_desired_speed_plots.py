import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval

df = pd.DataFrame()
filename = "experiments/data/Exp_Stress_DistinctSamples20_MaxSteps10000_Repi10.csv"
df = df.append(pd.read_csv(filename, converters={'Panic level': literal_eval}))


# Plot evacuation time
x = df.groupby("prob_stressed").mean().reset_index()["prob_stressed"]
y = df.groupby("prob_stressed").mean()["Evacuation time"]
err = (1.96 * df.groupby("prob_stressed")["Evacuation time"].std()) / np.sqrt(10)

plt.plot(x, y, c='k')
plt.fill_between(x, y - err, y + err, alpha=0.7)
plt.xlabel("Probability Stressed")
plt.ylabel("Evacuation Time (s)")
plt.title("Evacuation Time based on Percetnage of Stressed People")
plt.savefig("experiments/images/exp_desired_speed_evactime.png")
plt.show()

# Plot flow
x = df.groupby("prob_stressed").mean().reset_index()["prob_stressed"]
y = df.groupby("prob_stressed").mean()["Flow"]
err = (1.96 * df.groupby("prob_stressed")["Flow"].std()) / np.sqrt(10)

plt.plot(x, y, c='k')
plt.fill_between(x, y - err, y + err, alpha=0.7)
plt.xlabel("Probability Stressed")
plt.ylabel("Flow (s)")
plt.title("OutfLow based on Percetnage of Stressed People")
plt.savefig("experiments/images/exp_desired_speed_flow.png")
plt.show()

# Plot Std exit times
x = df.groupby("prob_stressed").mean().reset_index()["prob_stressed"]
y = df.groupby("prob_stressed").mean()["std exit time"]
err = (1.96 * df.groupby("prob_stressed")["std exit time"].std()) / np.sqrt(10)

plt.plot(x, y, c='k')
plt.fill_between(x, y - err, y + err, alpha=0.7)
plt.xlabel("Probability Stressed")
plt.ylabel("Standard Deviation Exit Times (s)")
plt.title("Standard Deviation of the Exit Times Based on Percentage of Stressed People")
plt.savefig("experiments/images/exp_desired_speed_std.png")
plt.show()

# Plot panic
panic_data = df.sort_values(by=["prob_stressed"])

for i in range(1, 10):
    slice = panic_data[i * 10 - 10: i * 10]

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
plt.savefig("experiments/images/panic_plot.png")
