import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataframe_image as dfi

# Import train dataset
dataset = pd.read_csv("train.csv", index_col=0)
corr = dataset.corr(numeric_only=True)
# Styling and saving the correlation matrix
styled = corr.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1)
dfi.export(styled, "Correlation_matrix.png")
# Looking for missing values
print(dataset.isnull().sum())
