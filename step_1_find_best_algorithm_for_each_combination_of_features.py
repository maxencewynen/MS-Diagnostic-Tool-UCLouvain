import pandas as pd
import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, balanced_accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from itertools import chain, combinations
import warnings
from utils import *
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import numpy as np

global n_model
n_model = 0
np.random.seed(RANDOM_SEED)


def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))


def find_best_threshold(y_true, y_pred_proba):
    """
    Find the best threshold for the given predictions
    :param y_true: GT labels
    :param y_pred_proba: predicted probabilities
    :return: best threshold
    """
    thresholds = np.linspace(0, 1, 100)
    best_threshold = None
    best_balanced_accuracy = -1
    best_sensitivity = -1

    for threshold in thresholds:
        y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        balanced_accuracy = (sensitivity + specificity) / 2

        if balanced_accuracy > best_balanced_accuracy or \
                (balanced_accuracy == best_balanced_accuracy and sensitivity > best_sensitivity):
            best_threshold = threshold
            best_balanced_accuracy = balanced_accuracy
            best_sensitivity = sensitivity

    return best_threshold


def cross_validate_best_model(X, y):
    models = [
        LogisticRegression(),
        SVC(probability=True),
        RandomForestClassifier(),
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        XGBClassifier(),
        AdaBoostClassifier(),
    ]

    best_model = None
    best_balanced_accuracy = -1
    associated_probability_threshold = None

    for model_class in models:
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
        # skf = KFold(n_splits=10, shuffle=True, random_state=42)
        balanced_accuracies = []
        probability_thresholds = []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[X_train.index], y[X_test.index]

            if USE_PIPELINE:
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", model_class)
                ])
            else:
                model = model_class

            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)
            best_threshold = find_best_threshold(y_test, y_pred_proba)
            y_pred = (y_pred_proba[:, 1] >= best_threshold).astype(int)
            balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
            probability_thresholds.append(best_threshold)

        avg_balanced_accuracy = np.mean(balanced_accuracies)

        if avg_balanced_accuracy > best_balanced_accuracy:
            best_model = model_class
            best_balanced_accuracy = avg_balanced_accuracy
            associated_probability_threshold = np.mean(probability_thresholds)

    best_model = str(best_model).split("(")[0]
    best_model = 'XGBClassifier()' if "XGBClassifier" in str(best_model) else best_model

    return best_model, associated_probability_threshold


def find_best_algorithm_for_specific_combination_of_features(X, y, features):
    global n_model
    n_model += 1
    feats = [f for f in features if f not in ("Select3*-v2NA", "Select6*-v2NA")]
    if "Select3*-v2NA" in features:
        feats += ["Select3*-v2NA_0.0", "Select3*-v2NA_1.0", "Select3*-v2NA_NA"]
    if "Select6*-v2NA" in features:
        feats += ["Select6*-v2NA_0.0", "Select6*-v2NA_1.0", "Select6*-v2NA_NA"]
    this_X = X[feats].dropna()
    this_y = y[this_X.index]

    # # #reindex
    # # this_X = this_X.reset_index(drop=True)
    # # this_y = this_y.reset_index(drop=True)
    # this_X, this_y = prepare_data(this_X, this_y)

    best_model, best_threshold = cross_validate_best_model(this_X, this_y)

    return best_model, best_threshold


def make_combinations(excluded_features):
    all_features = [f for f in ALL_FEATURES if f not in excluded_features]
    all_combinations = powerset(all_features)
    n_combinations = 2 ** len(all_features) - 1
    print(f"Number of combinations: {n_combinations}")

    # remove all combinations where both Select3*-v2NA and % perivenular les are present
    all_combinations = [c for c in all_combinations if not ("Select3*-v2NA" in c and "% perivenular les" in c)]
    # remove all combinations where both Select6*-v2NA and % perivenular les are present
    all_combinations = [c for c in all_combinations if not ("Select6*-v2NA" in c and "% perivenular les" in c)]
    # remove all combinations where both PRL1 and number_PRL are present
    all_combinations = [c for c in all_combinations if not ("PRL1" in c and "number_PRL" in c)]
    # remove all combinations where both CL1 and CL-count-updated are present
    all_combinations = [c for c in all_combinations if not ("CL1" in c and "CL-count-updated" in c)]
    # remove all combinations where both Select3*-v2NA and Select6*-v2NA are present
    all_combinations = [c for c in all_combinations if not ("Select3*-v2NA" in c and "Select6*-v2NA" in c)]

    print(f"Number of combinations after filtering: {len(all_combinations)}")

    return all_combinations


def find_best_algorithm_for_each_combination(X, y):
    excluded_features = ("age", "sex", "Filippi", "OCB_presence")
    all_features = [f for f in ALL_FEATURES if f not in excluded_features]
    all_combinations = make_combinations(excluded_features)
    best_model_per_combination = []

    for i, combination in enumerate(all_combinations):
        print(f"Progress: {i + 1}/{len(all_combinations)}")
        best_model, best_threshold = find_best_algorithm_for_specific_combination_of_features(X, y, combination)
        print(f"Best model for combination {combination}: {best_model}, best threshold: {best_threshold}\n")

        feature_presence = {f: f in list(combination) for f in sorted(all_features, key=lambda x: x.upper())}
        infos = {
            "model": best_model,
            "threshold": best_threshold,
        }
        infos.update(feature_presence)
        best_model_per_combination.append(infos)

    best_model_per_combination = pd.DataFrame(best_model_per_combination)

    return best_model_per_combination


if __name__ == "__main__":
    X, y = get_full_dataset(TRAINING_DATA_PATH, dropna=False, return_X_y=True, prepare=True)
    best_models = find_best_algorithm_for_each_combination(X, y)
    best_models.to_csv(BEST_ALGORITHM_PER_COMBINATION_OF_FEATURES_PATH, index=False)
    print("Done.")
