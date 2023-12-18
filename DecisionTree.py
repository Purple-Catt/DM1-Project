import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, RocCurveDisplay
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import sklearn
import warnings
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
warnings.simplefilter("ignore", sklearn.exceptions.UndefinedMetricWarning)
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
scale = True
plot = True
save = False
which = 2
#scores = []

train_df = pd.read_csv("TRAIN_DF.csv", index_col=0)
test_df = pd.read_csv("TEST_DF.csv", index_col=0)

y_train = train_df.pop("genre")
y_test = test_df.pop("genre")
if scale:
    scaler = StandardScaler()
    x_train = pd.DataFrame(data=scaler.fit_transform(train_df), columns=train_df.columns)
    x_test = pd.DataFrame(data=scaler.fit_transform(test_df), columns=test_df.columns)

else:
    x_train = train_df.copy(deep=True)
    x_test = test_df.copy(deep=True)


def gridsearch():
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(),
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
    results["metric_weight"] = results["param_criterion"]
    sns.lineplot(
        data=results, x="param_max_depth", y="mean_test_score", hue="metric_weight"
    )
    plt.show()

    # Utilizza i risultati migliori per migliorare il classificatore DecisionTree
    dt = DecisionTreeClassifier(**grid.best_params_)
    dt.fit(x_train, y_train)
    y_test_pred = dt.predict(x_test)
    y_train_pred = dt.predict(x_train)
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
    y_test_prob = dt.predict_proba(x_test)

    if which == 0:
        RocCurveDisplay.from_predictions(
            y_test_bin[:, class_id],
            y_test_prob[:, class_id],
            name=f"{class_of_interest} vs the rest",
            color="darkorange",
            plot_chance_level=True,
        )
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
    if which == 0:
        plt.title("One-vs-Rest ROC curves")
    elif which == 1:
        plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
    elif which == 2:
        plt.title("Decision Tree\nMicro-averaged One-vs-Rest ROC")
    plt.legend()
    plt.show()


def random_forest():
    param_grid = {
        "n_estimators": [100, 200, 300],
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    grid = GridSearchCV(
        RandomForestClassifier(),
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
    results["metric_weight"] = results["param_criterion"]
    sns.lineplot(
        data=results, x="param_max_depth", y="mean_test_score", hue="metric_weight"
    )
    plt.show()

    # Utilizza i risultati migliori per migliorare il classificatore Random Forest
    rf = RandomForestClassifier(**grid.best_params_)
    rf.fit(x_train, y_train)
    y_test_pred = rf.predict(x_test)
    y_train_pred = rf.predict(x_train)
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
        ).transpose().to_csv("class_report_test_rf.csv")
        pd.DataFrame.from_dict(
            classification_report(y_train, y_train_pred, output_dict=True)
        ).transpose().to_csv("class_report_train_rf.csv")

    lb = LabelBinarizer()
    lb.fit(y_train)
    y_test_bin = lb.transform(y_test)
    class_of_interest = 18
    class_id = np.flatnonzero(lb.classes_ == class_of_interest)[0]
    y_test_prob = rf.predict_proba(x_test)

    if which == 0:
        RocCurveDisplay.from_predictions(
            y_test_bin[:, class_id],
            y_test_prob[:, class_id],
            name=f"{class_of_interest} vs the rest",
            color="darkorange",
            plot_chance_level=True,
        )
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
    if which == 0:
        plt.title("One-vs-Rest ROC curves")
    elif which == 1:
        plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
    elif which == 2:
        plt.title("Random Forest\nMicro-averaged One-vs-Rest ROC")
    plt.legend()
    plt.show()


def adaboost():
    param_grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.1, 0.5, 1.0],
    }

    grid = GridSearchCV(
        AdaBoostClassifier(),
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
    results["metric_weight"] = results["param_learning_rate"]
    sns.lineplot(
        data=results, x="param_n_estimators", y="mean_test_score", hue="metric_weight"
    )
    plt.show()

    # Utilizza i risultati migliori per migliorare il classificatore AdaBoost
    ab = AdaBoostClassifier(**grid.best_params_)
    ab.fit(x_train, y_train)
    y_test_pred = ab.predict(x_test)
    y_train_pred = ab.predict(x_train)
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
        ).transpose().to_csv("class_report_test_ab.csv")
        pd.DataFrame.from_dict(
            classification_report(y_train, y_train_pred, output_dict=True)
        ).transpose().to_csv("class_report_train_ab.csv")

    lb = LabelBinarizer()
    lb.fit(y_train)
    y_test_bin = lb.transform(y_test)
    class_of_interest = 18
    class_id = np.flatnonzero(lb.classes_ == class_of_interest)[0]
    y_test_prob = ab.predict_proba(x_test)

    if which == 0:
        RocCurveDisplay.from_predictions(
            y_test_bin[:, class_id],
            y_test_prob[:, class_id],
            name=f"{class_of_interest} vs the rest",
            color="darkorange",
            plot_chance_level=True,
        )
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
    if which == 0:
        plt.title("One-vs-Rest ROC curves")
    elif which == 1:
        plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
    elif which == 2:
        plt.title("AdaBoost\nMicro-averaged One-vs-Rest ROC")
    plt.legend()
    plt.show()

def cross_val():
    scores = {}  # Initialize an empty dictionary to store the scores
    for n in range(80, 200, 10):
        dt = DecisionTreeClassifier()
        cross_v = cross_val_score(estimator=dt, X=x_train, y=y_train, cv=5)
        scores[str(n)] = cross_v.mean()  # Use string keys instead of integer keys

    # Perform grid search
    gridsearch()
    # Print classification report
    dt_best = DecisionTreeClassifier()  # Replace with your best estimator from grid search
    dt_best.fit(x_train, y_train)
    y_pred = dt_best.predict(x_test)
    report = classification_report(y_test, y_pred, digits=4)  # Set digits parameter to 4
    print(report)
    attributes = [col for col in train_df.columns]
    plt.figure(figsize=(20, 4), dpi=300)
    plot_tree(dt_best, feature_names=attributes, filled=True)
    plt.show()

    random_forest()
    # Print classification report
    dt_best = DecisionTreeClassifier()  # Replace with your best estimator from grid search
    dt_best.fit(x_train, y_train)
    y_pred = dt_best.predict(x_test)
    report = classification_report(y_test, y_pred, digits=4)  # Set digits parameter to 4
    print(report)
    attributes = [col for col in train_df.columns]
    plt.figure(figsize=(20, 4), dpi=300)
    plot_tree(dt_best, feature_names=attributes, filled=True)
    plt.show()

    adaboost()
    # Print classification report
    dt_best = DecisionTreeClassifier()  # Replace with your best estimator from grid search
    dt_best.fit(x_train, y_train)
    y_pred = dt_best.predict(x_test)
    report = classification_report(y_test, y_pred, digits=4)  # Set digits parameter to 4
    print(report)
    attributes = [col for col in train_df.columns]
    plt.figure(figsize=(20, 4), dpi=300)
    plot_tree(dt_best, feature_names=attributes, filled=True)
    plt.show()

cross_val()
