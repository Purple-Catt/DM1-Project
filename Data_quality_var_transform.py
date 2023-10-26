import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import math
import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
dataset = pd.read_csv("train.csv")
test_dataset = pd.read_csv("test.csv")
drop_label = ["features_duration_ms", "n_beats", "n_bars", "energy"]


def missing_bar():
    plt.bar(x=["Mode", "Time signature", "Popularity confidence"],
            height=dataset[["mode", "time_signature", "popularity_confidence"]].isna().sum() * 100 / len(dataset),
            width=0.5)
    plt.title(label="Missing values", fontdict={"fontsize": "18", "weight": "bold"})
    plt.xlabel("Attributes", fontdict={"fontsize": "12"})
    plt.ylabel("None values (%)", fontdict={"fontsize": "12"})
    # plt.xticks(rotation=45)
    # plt.savefig("Data quality and transformations\\Missing_values.png")


num_cols = ["duration_ms", "popularity", "danceability", "key", "loudness", "mode",
            "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo",
            "time_signature", "processing"]


def missing_heatmap():
    dataset.drop(columns=drop_label, inplace=True)
    cols = dataset.columns
    for i in dataset.index:
        if dataset["time_signature"].iloc[i] == 0:
            dataset["time_signature"].iloc[i] = np.NaN
    colours = ["#fbbd3c", "#2574f4"]
    sns.heatmap(dataset[cols].isnull(), cmap=sns.color_palette(colours), yticklabels=False)
    plt.tight_layout()
    plt.show()


def mode_rep():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.bar(x=["0", "1"], height=dataset["mode"].value_counts(), color="#fbbd3c")
    ax1.set_ylim(0, 10000)
    ax1.set_title("Before replacing")
    filling = dataset["mode"].value_counts(normalize=True)
    missing = dataset["mode"].isnull()
    dataset.loc[missing, "mode"] = np.random.choice(filling.index, size=len(dataset[missing]), p=filling.values)
    ax2.bar(x=["0", "1"], height=dataset["mode"].value_counts(), color="#2574f4")
    ax2.set_ylim(0, 10000)
    ax2.set_title("After replacing")
    plt.show()


def time_sig_rep():
    for i in dataset.index:
        if dataset["time_signature"].iloc[i] == 0:
            dataset["time_signature"].iloc[i] = np.NaN
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    val = dataset["time_signature"].value_counts()
    val._set_value(2.0, 0)
    val = val.reindex([1.0, 2.0, 3.0, 4.0, 5.0])
    ax1.bar(x=[1, 2, 3, 4, 5], height=val, color="#fbbd3c")
    ax1.set_ylim(0, 14000)
    ax1.set_title("Before replacing")
    filling = dataset["time_signature"].value_counts(normalize=True)
    missing = dataset["time_signature"].isnull()
    dataset.loc[missing, "time_signature"] = np.random.choice(filling.index,
                                                              size=len(dataset[missing]), p=filling.values)
    val = dataset["time_signature"].value_counts()
    val._set_value(2.0, 0)
    val = val.reindex([1.0, 2.0, 3.0, 4.0, 5.0])
    ax2.bar(x=["1", "2", "3", "4", "5"], height=val, color="#2574f4")
    ax2.set_ylim(0, 14000)
    ax2.set_title("After replacing")
    plt.show()


def duplicates():
    duplicate = dataset["name"].value_counts() > 1
    for i in duplicate:
        if i is True:
            print(dataset.iloc[i, :])


def pearson_corr_property():
    mean1 = dataset["duration_ms"].mean()
    mean2 = dataset["n_beats"].mean()
    std1 = dataset["duration_ms"].std()
    std2 = dataset["n_beats"].std()
    dist1 = (dataset["duration_ms"] - mean1) / std1
    dist2 = (dataset["n_beats"] - mean2) / std2
    print(math.dist(dataset["duration_ms"], dataset["n_beats"]))
    print(math.dist(dist1, dist2))
    print(dataset["duration_ms"].corr(dataset["n_beats"]))
    print(dist1.corr(dist2))


def tempo_standardization():
    tempo_mean = dataset["tempo"].mean()
    tempo_std = dataset["tempo"].std()

    tempo_dist = (dataset["tempo"] - tempo_mean) / tempo_std
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(dataset["tempo"])
    ax2.hist(tempo_dist)
    plt.show()


def autoencoder_NN():
    dataset.drop(columns=drop_label, inplace=True)
    for feature in ["duration_ms", "loudness"]:
        encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1)  # one-dimensional output
        ])

        decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(2)  # decode to two dimensions again
        ])

        autoencoder = tf.keras.Sequential([
            encoder,
            decoder
        ])

        autoencoder.compile(loss="mse")
        autoencoder.fit(
            x=dataset[feature],  # goal is that output is
            y=dataset[feature],  # close to the same input
            validation_split=0.2,  # to check if the model is generalizing
            epochs=100)

        data_tensor = tf.convert_to_tensor(dataset[[feature]]).numpy()
        reconstructed_points = autoencoder(data_tensor)
        reconstruction_error = tf.reduce_sum(
            ((reconstructed_points - data_tensor) ** 2), axis=1
        )  # MSE
        df = pd.DataFrame(
            {"x": dataset[feature].values,
             "y": dataset.index,
             "reconstruction_error": reconstruction_error}
        )
        max_mae = np.percentile(a=df["reconstruction_error"], q=99.8)
        sns.lineplot(df["x"])
        sns.scatterplot(df["x"].loc[df["reconstruction_error"] > max_mae], color="r")
        plt.xlabel("Object N°")
        plt.ylabel("Loudness (dB)")
        plt.title("Outlier loudness")
        plt.show()
