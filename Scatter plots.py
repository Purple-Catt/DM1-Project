import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("train.csv", index_col=0)
# After looking at the correlation matrix, these plots help catching
# the correlation between the variables that are relevant
plt.scatter(dataset["duration_ms"], dataset["features_duration_ms"])
plt.xlabel("Duration ms")
plt.ylabel("Features duration ms")
plt.savefig("Scatter_plot\\duration-features_duration scatterplot.png")
plt.clf()

plt.scatter(dataset["n_beats"], dataset["n_bars"])
plt.xlabel("N beats")
plt.ylabel("N bars")
plt.savefig("Scatter_plot\\Beats-bars scatterplot.png")
plt.clf()

plt.scatter(dataset["n_beats"], dataset["duration_ms"])
plt.xlabel("N beats")
plt.ylabel("Duration ms")
plt.savefig("Scatter_plot\\Beats-duration scatterplot.png")
plt.clf()

plt.scatter(dataset["n_bars"], dataset["duration_ms"])
plt.xlabel("N bars")
plt.ylabel("Duration ms")
plt.savefig("Scatter_plot\\Bars-duration scatterplot.png")
plt.clf()