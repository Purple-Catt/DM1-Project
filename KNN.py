import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, RocCurveDisplay
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import sklearn
import warnings

warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
warnings.simplefilter("ignore", sklearn.exceptions.UndefinedMetricWarning)
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
scale = True
plot = False
save = False
which = 2
scores = []

train_df = pd.read_csv("Datasets/TRAIN_DF.csv", index_col=0)
test_df = pd.read_csv("Datasets/TEST_DF.csv", index_col=0)

y_train = train_df.pop("genre")
y_test = test_df.pop("genre")
if scale:
    scaler = StandardScaler()
    x_train = pd.DataFrame(data=scaler.fit_transform(train_df), columns=train_df.columns)
    x_test = pd.DataFrame(data=scaler.fit_transform(test_df), columns=test_df.columns)

else:
    x_train = train_df.copy(deep=True)
    x_test = test_df.copy(deep=True)


def knn_eval():
    knn = KNeighborsClassifier(n_neighbors=33, metric="cityblock", weights="distance")
    knn.fit(x_train, y_train)
    y_test_pred = knn.predict(x_test)
    y_train_pred = knn.predict(x_train)
    if plot:
        cf = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cf, annot=True, cmap="Greens")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.tight_layout()
        plt.show()

    if save:
        pd.DataFrame.from_dict(
            classification_report(y_test, y_test_pred, output_dict=True)
        ).transpose().to_csv("class_report_test.csv")
        pd.DataFrame.from_dict(
            classification_report(y_train, y_train_pred, output_dict=True)
        ).transpose().to_csv("class_report_train.csv")

    lb = LabelBinarizer()
    lb.fit(y_train)
    y_test_bin = lb.transform(y_test)
    class_of_interest = 18
    class_id = np.flatnonzero(lb.classes_ == class_of_interest)[0]
    y_test_prob = knn.predict_proba(x_test)

    if which == 0:
        RocCurveDisplay.from_predictions(
            y_test_bin[:, class_id],
            y_test_prob[:, class_id],
            name=f"{class_of_interest} vs the rest",
            color="darkorange",
            plot_chance_level=True,
        )
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("One-vs-Rest ROC curves")
        plt.legend()
        plt.show()

    elif which == 1:
        fig, ax = plt.subplots(figsize=(6, 6))
        for class_id in range(0, 20):
            RocCurveDisplay.from_predictions(
                y_test_bin[:, class_id],
                y_test_prob[:, class_id],
                name=f"ROC curve for {class_id}",
                ax=ax,
                plot_chance_level=(class_id == 2),
            )

        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
        plt.legend()
        plt.show()

    elif which == 2:
        RocCurveDisplay.from_predictions(
            y_test_bin.ravel(),
            y_test_prob.ravel(),
            name="micro-average OvR",
            color="darkorange",
            plot_chance_level=True,
        )
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("KNN\nMicro-averaged One-vs-Rest ROC")
        plt.legend()
        plt.show()


def cross_val():
    for n in range(80, 200, 10):
        knn = KNeighborsClassifier(n_neighbors=n, metric="euclidean")
        cross_v = cross_val_score(estimator=knn, X=x_train, y=y_train, cv=5)
        scores[n] = cross_v.mean()


def gridsearch():
    param_grid = {
        "n_neighbors": np.arange(5, 150, 2),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "cityblock"],
    }

    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid=param_grid,
        cv=RepeatedStratifiedKFold(random_state=0, n_repeats=5),
        n_jobs=-1,
        refit=True,
        verbose=4
    )

    grid.fit(x_train, y_train)
    clf = grid.best_estimator_

    print(grid.best_params_, grid.best_score_)

    y_test_pred = clf.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, y_test_pred))

    clf.score(x_test, y_test)

    results = pd.DataFrame(grid.cv_results_)
    results["metric_weight"] = results["param_metric"] + ", " + results["param_weights"]
    sns.lineplot(
        data=results, x="param_n_neighbors", y="mean_test_score", hue="metric_weight"
    )
    plt.show()


knn_eval()
