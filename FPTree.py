from fim import fpgrowth
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import seaborn as sns

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]

supp_plot = False
heatmap_plot = False
genre_dict = {0: "afrobeat", 1: "black-metal", 2: "bluegrass", 3: "brazil", 4: "breakbeat",
              5: "chicago-house", 6: "disney", 7: "forro", 8: "happy", 9: "idm",
              10: "indian", 11: "industrial", 12: "iranian", 13: "j-dance",
              14: "j-dol", 15: "mandopop", 16: "sleep", 17: "spanish", 18: "study", 19: "techno"}
df = pd.read_csv("Datasets/TRAIN_DF.csv", index_col=0)
for var in ["duration_ms", "popularity", "danceability", "loudness",
            "speechiness", "acousticness", "instrumentalness",
            "liveness", "valence", "tempo"]:
    newf = var + "Bin"
    df[newf] = pd.qcut(df[var], 3)
    df.drop([var], axis=1, inplace=True)
    df[newf] = df[newf].astype(str) + f"_{var}"

df["genre"] = df["genre"].map(genre_dict)

supp = 10
zmin = 2
X = df.values.tolist()

itemsets = fpgrowth(X, target="s", supp=supp, zmin=zmin, report="S")
fp_df = pd.DataFrame(itemsets, columns=["frequent_itemset", "support"])
print(fp_df)

conf = 70
rules = fpgrowth(X, target="r", supp=supp, zmin=zmin, conf=conf, report="aScl")
print(pd.DataFrame(
    rules,
    columns=[
        "consequent",
        "antecedent",
        "abs_support",
        "%_support",
        "confidence",
        "lift",
    ],
))

if supp_plot:
    len_max_it = []
    len_cl_it = []
    max_supp = 25
    for i in range(2, max_supp):
        max_itemsets = fpgrowth(X, target="m", supp=i, zmin=zmin)
        cl_itemsets = fpgrowth(X, target="c", supp=i, zmin=zmin)
        len_max_it.append(len(max_itemsets))
        len_cl_it.append(len(cl_itemsets))

    plt.plot(np.arange(2, max_supp), len_max_it, label="maximal")
    plt.plot(np.arange(2, max_supp), len_cl_it, label="closed")
    plt.legend()
    plt.xlabel("%support")
    plt.ylabel("itemsets")

    plt.show()
    plt.clf()

if heatmap_plot:
    len_r = []
    min_sup = 1
    max_sup = 20
    min_conf = 50
    max_conf = 90
    for i in range(min_sup, max_sup):  # support
        len_r_wrt_i = []
        for j in range(min_conf, max_conf):  # confidence
            rules = fpgrowth(X, target="r", supp=i, zmin=zmin, conf=j, report="aScl")
            len_r_wrt_i.append(len(rules))  # study your characteristics/properties here

        len_r.append(len_r_wrt_i)
    len_r = np.array(len_r)

    sns.heatmap(len_r, cmap="Blues", fmt='g')
    plt.yticks(np.arange(0, max_sup - min_sup + 1, 5), np.arange(min_sup, max_sup + 1, 5))
    plt.xticks(np.arange(0, max_conf - min_conf + 1, 5), np.arange(min_conf, max_conf + 1, 5))
    plt.xlabel("%confidence")
    plt.ylabel("%support")
    plt.show()
