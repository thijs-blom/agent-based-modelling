import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

files = ["1","2","3","4","5"]
df = pd.DataFrame()

# Read all data files
for iteration in files:
    filename=f"SA_Data/Validation_DistinctSamples10_MaxSteps100000_Repi1_{iteration}.csv"
    df = df.append(pd.read_csv(filename))
    filename=f"SA_Data/Validation_DistinctSamples9_MaxSteps100000_Repi1_{iteration}.csv"
    df = df.append(pd.read_csv(filename))
    filename=f"SA_Data/Validation_DistinctSamples19_MaxSteps100000_Repi1_{iteration}.csv"
    df = df.append(pd.read_csv(filename))


x = df.groupby("init_desired_speed").mean().reset_index()["init_desired_speed"]
y = df.groupby("init_desired_speed").mean()["Flow / Desired Velocity"]
err = (1.96 * df.groupby("init_desired_speed")["Flow / Desired Velocity"].std()) / np.sqrt(len(files))

plt.plot(x, y, c='k')
plt.fill_between(x, y - err, y + err, alpha=0.7)
plt.xlabel("Desired Velocity (m/s)")
plt.ylabel(r"Pedestrian Flow (m$^{-1}$s$^{-1}$) / Desired Velocity (m/s)")
plt.title("Pedestrian Flow Compared to Desired Velocity")
plt.savefig("validation_all.png")
plt.show()
