import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import Data_quality_var_transform as dq
import warnings

warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)

train_raw = pd.read_csv("train.csv")
dataset = pd.read_csv("df_cleaned.csv", index_col=0)
cl_col = ["duration_ms", "popularity", "danceability", "loudness", "speechiness",
          "acousticness", "instrumentalness", "liveness", "valence", "tempo", "genre"]
train_df = dataset[cl_col].copy(deep=True)
test_dataset = pd.read_csv("test.csv")

duration = train_raw["duration_ms"].iloc[train_df.index]
train_mean = duration.mean()
train_std = duration.std()
dq.var_transformation(duration_mean=train_mean, duration_std=train_std, dataset=test_dataset)
test_df = test_dataset[cl_col].copy(deep=True)

train_df.to_csv("TRAIN_DF.csv")
test_df.to_csv("TEST_DF.csv")
