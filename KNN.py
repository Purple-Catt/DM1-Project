import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings

warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
warnings.simplefilter("ignore", sklearn.exceptions.UndefinedMetricWarning)

train_df = pd.read_csv("TRAIN_DF.csv", index_col=0)
test_df = pd.read_csv("TEST_DF.csv", index_col=0)

y_train = train_df.pop("genre")
x_train = train_df.copy(deep=True)
y_test = test_df.pop("genre")
x_test = test_df.copy(deep=True)

for n in [121]:
    knn = KNeighborsClassifier(n_neighbors=n, metric="euclidean")
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    cf = confusion_matrix(y_test, y_pred)
    sns.heatmap(cf, annot=True, cmap="Greens")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.show()


def cross_val():
    knn = KNeighborsClassifier(n_neighbors=121, metric="euclidean")
    cross_v = cross_validate(estimator=knn, X=x_train, y=y_train, cv=5)
    print(cross_v)
