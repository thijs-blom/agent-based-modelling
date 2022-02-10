from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read all data files
df = pd.DataFrame()
for file in (Path(__file__).parent / "data").iterdir():
    df = df.append(pd.read_csv(file))

x = df.groupby("init_desired_speed").mean().reset_index()["init_desired_speed"]
y = df.groupby("init_desired_speed").mean()["Flow / Desired Velocity"]
err = (1.96 * df.groupby("init_desired_speed")["Flow / Desired Velocity"].std()) / np.sqrt(10)

plt.plot(x, y, color='k', label="mean")
plt.fill_between(x, y - err, y + err, color='pink', label="confidence interval")
plt.xlabel("Desired Velocity (m/s)")
plt.ylabel(r"Pedestrian Flow (m$^{-1}$s$^{-1}$) / Desired Velocity (m/s)")
plt.title("Pedestrian Flow Compared to Desired Velocity")
plt.legend()
plt.savefig(Path(__file__).parent / "images/validation_all.png")
plt.show()

plt.clf()

y = df.groupby("init_desired_speed").mean()["Evacuation time"]
err = (1.96 * df.groupby("init_desired_speed")["Evacuation time"].std()) / np.sqrt(10)

plt.plot(x, y, color='k', label="mean")
plt.fill_between(x, y - err, y + err, color='pink', label="confidence interval")
plt.xlabel("Desired Velocity (m/s)")
plt.ylabel("Evacuation Time (s)")
plt.title("Evacuation Time Compared to Desired Velocity")
plt.legend()
plt.savefig(Path(__file__).parent / "images/validation_exit_time.png")
plt.show()
