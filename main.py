import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataframe_image as dfi
# Import train dataset
dataset = pd.read_csv("train.csv")
corr = dataset.corr(numeric_only=True)
# Styling and saving the correlation matrix
styled = corr.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1)
# dfi.export(styled, "Correlation_matrix.png")
# Looking for missing values
# print(dataset.isnull().sum())
# dataset["genre"].value_counts().plot(kind="bar", title="Genre", ylim=(0, 1000))
# plt.savefig("Distribution_plot\\genre.png")
outliers_over = np.percentile(dataset["duration_ms"], q=90)
dataset.drop(axis="index", index=dataset.loc[dataset["duration_ms"] >= outliers_over, "duration_ms"].index)
