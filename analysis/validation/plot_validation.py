import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

files = ["1", "2", "3", "4", "5"]
df = pd.DataFrame()

# Read all data files
for iteration in files:
    filename = f"analysis/validation/data/Validation_DistinctSamples10_MaxSteps100000_Repi1_{iteration}.csv"
    df = df.append(pd.read_csv(filename))
    filename = f"analysis/validation/data/Validation_DistinctSamples9_MaxSteps100000_Repi1_{iteration}.csv"
    df = df.append(pd.read_csv(filename))
    filename = f"analysis/validation/data/Validation_DistinctSamples19_MaxSteps100000_Repi1_{iteration}.csv"
    df = df.append(pd.read_csv(filename))

x = df.groupby("init_desired_speed").mean().reset_index()["init_desired_speed"]
y = df.groupby("init_desired_speed").mean()["Flow / Desired Velocity"]
err = (1.96 * df.groupby("init_desired_speed")["Flow / Desired Velocity"].std()) / np.sqrt(10)

plt.plot(x, y, color='k', label="mean")
plt.fill_between(x, y - err, y + err, color='pink', label="confidence interval")
plt.xlabel("Desired Velocity (m/s)")
plt.ylabel(r"Pedestrian Flow (m$^{-1}$s$^{-1}$) / Desired Velocity (m/s)")
plt.title("Pedestrian Flow Compared to Desired Velocity")
plt.legend()
plt.savefig("analysis/validation/images/validation_all.png")
plt.show()

y = df.groupby("init_desired_speed").mean()["Evacuation time"]
err = (1.96 * df.groupby("init_desired_speed")["Evacuation time"].std()) / np.sqrt(10)

plt.plot(x, y, color='k', label="mean")
plt.fill_between(x, y - err, y + err, color='pink', label="confidence interval")
plt.xlabel("Desired Velocity (m/s)")
plt.ylabel("Evacuation Time (s)")
plt.title("Evacuation Time Compared to Desired Velocity")
plt.legend()
plt.savefig("analysis/validation/images/validation_exit_time.png")
plt.show()
