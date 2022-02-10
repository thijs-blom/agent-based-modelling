import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval
from pathlib import Path
import scipy.optimize 

df = pd.DataFrame()
filename = Path(__file__).parent / "data/Exp_Prob_DistinctSamples10_MaxSteps10000_Repi5_Vision2_pop200.csv"
df = df.append(pd.read_csv(filename, converters={'strategy_weights': literal_eval}))

# Group by probability of nearest exit
df["strategy_weights"] = df["strategy_weights"].apply(lambda x: list(x)[0])
x = df.groupby("strategy_weights").mean().reset_index()["strategy_weights"]

# Plot Flow
y = df.groupby("strategy_weights").mean()["Flow"]
err = (1.96 * df.groupby("strategy_weights")["Flow"].std()) / np.sqrt(5)

plt.figure(figsize=(6,4),dpi=300)
plt.errorbar(x, y, yerr=err, fmt='o', color='darkgreen',
            ecolor='green', elinewidth=2, capsize=0)
plt.ylabel("Pedestrian Flow (m$^{-1}$s$^{-1}$)", fontsize=12)
plt.xlabel("Probability of 'Nearest Exit' %", fontsize=12)
plt.title("'Follow the Leader' and 'Nearest Exit' Strategy Mixing Experiment", fontsize=12)
plt.xticks(fontsize= 10)
plt.yticks(fontsize= 10)
plt.savefig(Path(__file__).parent / "images/exp_Prob_follow_flow.png")


plt.clf()
# Plot Evacuation Percentage

y = df.groupby("strategy_weights").mean()["Evacuation percentage"]
err = (1.96 * df.groupby("strategy_weights")["Evacuation percentage"].std()) / np.sqrt(5)

plt.figure(figsize=(6,4),dpi=300)
plt.errorbar(x, y, yerr=err, fmt='o', color='darkorange',
    ecolor='orange', elinewidth=2, capsize=0)
plt.xlabel("Probability of 'Nearest Exit' %", fontsize=12)
plt.ylabel(f"Evacuation Percentage (%)", fontsize=12)
plt.title("'Follow the Leader' and 'Nearest Exit' Strategy Mixing Experiment", fontsize=12)
plt.xticks(fontsize= 10)
plt.yticks(fontsize= 10)
plt.savefig(Path(__file__).parent / "images/exp_Prob_follow_evacperc.png")

# #plt.plot(x, y, c='k')
# plt.errorbar(x, y, yerr=err, fmt='o', color='darkblue', ecolor='blue', elinewidth=2, capsize=0)
# #plt.fill_between(x, y - err, y + err, alpha=0.7)
# plt.xlabel("Probability of Nearest Exit Strategy (compared to Follow the Leader)")
# plt.ylabel("Evacuation Time (s)")
# plt.title("Evacuation Time Nearest Exit vs. Follow the Leader Strategy")
# plt.savefig(Path(__file__).parent / "images/exp_Prob_follow_evactime.png")
# plt.show()

# plt.clf()

# y = df.groupby("strategy_weights").mean()["Evacuation percentage"]
# err = (1.96 * df.groupby("strategy_weights")["Evacuation percentage"].std()) / np.sqrt(5)


# plt.plot(x, y, c='k')
# #plt.errorbar(x, y, yerr=err, fmt='o', color='darkblue', ecolor='blue', elinewidth=2, capsize=0)
# plt.fill_between(x, y - err, y + err, alpha=0.7)
# plt.xlabel("Probability of Nearest Exit Strategy (compared to Follow the Leader)")
# plt.ylabel("Evacuation Percentage (%)")
# plt.title("Evacuation Percentage Nearest Exit vs. Follow the Leader Strategy")
# plt.savefig(Path(__file__).parent / "images/exp_Prob_follow_perc.png")
# plt.show()

# plt.clf()

# y = df.groupby("strategy_weights").mean()["Flow"]
# err = (1.96 * df.groupby("strategy_weights")["Flow"].std()) / np.sqrt(5)

# plt.plot(x, y, c='k')
# plt.fill_between(x, y - err, y + err, alpha=0.7)
# plt.xlabel("Probability Stressed")
# plt.ylabel("Flow (s)")
# plt.title("Outflow Nearest Exit vs. Follow the Leader Strategy")
# plt.savefig(Path(__file__).parent / "images/exp_Prob_follow_flow.png")
# plt.show()

# plt.clf()