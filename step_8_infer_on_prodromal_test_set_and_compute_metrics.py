"""Load models and run them on verona data + compute metrics."""

import os
import pickle
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utils import *
import xgboost as xgb
import numpy as np

# import warnings
# from sklearn.exceptions import InconsistentVersionWarning, UndefinedMetricWarning
# warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
# warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
pd.options.display.width = 0

PRODROMAL_TEST_DATA_PATH = "/home/mwynen/data/MSDT/test_set_prodromal.xlsx"
PRODROMAL_TEST_PREDICTIONS_PATH = "/home/mwynen/data/MSDT/test_set_prodromal_predictions.csv"

pprint("*** Step 8: Infer on the prodromal test set and compute metrics ***")
pprint()

def baseline_performance_on_the_whole_dataset():
    X, y = get_full_dataset(PRODROMAL_TEST_DATA_PATH, dropna=False, return_X_y=True, prepare=True)
    X = X.dropna(subset=["% perivenular les"])
    y = y[X.index]
    df = X
    df["diagnosis"] = y

    df["McDonald 2017"] = df["Thompson"] # & df["OCB_presence"]

    # Compute metrics
    balanced_acc_mcdonald = balanced_accuracy_score(df["diagnosis"], df["McDonald 2017"])
    conf_mat_mcdonald = confusion_matrix(df["diagnosis"], df["McDonald 2017"])
    tn, fp, fn, tp = conf_mat_mcdonald.ravel()
    precision_mcdonald = tp / (tp + fp + 1e-10)
    recall_mcdonald = tp / (tp + fn + 1e-10)
    sensitivity_mcdonald = tp / (tp + fn + 1e-10)
    specificity_mcdonald = tn / (tn + fp + 1e-10)
    f1_mcdonald = f1_score(df["diagnosis"], df["McDonald 2017"])

    pprint("Metrics for McDonald 2017:")
    pprint(f"Balanced Accuracy: {round(balanced_acc_mcdonald*100,1)}")
    pprint(f"Precision: {round(precision_mcdonald*100,1)}")
    pprint(f"Sensitivity: {round(sensitivity_mcdonald*100,1)}")
    pprint(f"Specificity: {round(specificity_mcdonald*100,1)}")
    pprint(f"F1 Score: {round(f1_mcdonald*100,1)}")
    pprint()


def get_features_from_model(model):
    if isinstance(model, xgb.XGBClassifier):
        return model.get_booster().feature_names
    else:
        return model.feature_names_in_


df_models = pd.read_csv(ALL_MODELS_FOR_ONLINE_TOOL_PATH)
test_results = []


def run_model(row, return_data_and_pred=False):
    model_name = row["model"]
    model_path = row["model_path"]
    print(f"Running model {model_path}...")
    model_threshold = row["threshold"]
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    # if row["% perivenular les"] and not row["CL1"] and not row["PRL1"] and not row["Thompson"] and not row["number_PRL"] and not row["CL-count-updated"]:
    #     print()

    features = get_features_from_model(model)

    X, y = get_full_dataset(PRODROMAL_TEST_DATA_PATH, dropna=False, return_X_y=True, prepare=True)

    # Drop nan
    X = X.dropna(subset=list(features) + ["% perivenular les"] if "% perivenular les" not in features else list(features))
    y = y[X.index]

    y_pred = model.predict_proba(X[features])[:, 1] > model_threshold
    y_true = y

    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sensitivity = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)

    metrics = ({
        "model": model_name,
        "f1": round(f1 * 100, 1),
        "precision": round(precision * 100, 1),
        "balanced_accuracy": round(balanced_accuracy * 100, 1),
        "sensitivity": round(sensitivity * 100, 1),
        "specificity": round(specificity * 100, 1)
    })

    if return_data_and_pred:
        # reassemble X, y_true, y_pred
        X = X.join(y_true)
        X["y_pred"] = y_pred
        return metrics, X
    return metrics, y_pred


for index, row in df_models.iterrows():
    out = run_model(row, return_data_and_pred=True)
    if out is not None:
        metrics, data_with_preds = out
    test_results.append(metrics)
    # break

data_with_preds.to_csv(PRODROMAL_TEST_PREDICTIONS_PATH, index=False)

df_results = pd.DataFrame(test_results)

def print_results_for_index(index):
    used_feats = [f for f in ALL_FEATURES if df_models[f].iloc[index]]
    pprint(f"Using the following features: {used_feats}")
    pprint(f"Metrics for model at index {index}:")
    pprint("Balanced Accuracy:", df_results["balanced_accuracy"].iloc[index])
    pprint("Precision:", df_results["precision"].iloc[index])
    pprint("Sensitivity:", df_results["sensitivity"].iloc[index])
    pprint("Specificity:", df_results["specificity"].iloc[index])
    pprint("F1 Score:", df_results["f1"].iloc[index])
    pprint()



baseline_performance_on_the_whole_dataset()

pprint("Metrics for best trained models:")
simplified_models_indices = df_models[df_models["simplified_variables_only"]].index
print_results_for_index(0)
pprint()
print_results_for_index(simplified_models_indices[0])
pprint()
print_results_for_index(21)

df_models_cvs = df_models[(df_models["% perivenular les"] & ~df_models["CL1"] & ~df_models["PRL1"] & ~df_models["Thompson"]
                           & ~df_models["number_PRL"] & ~df_models["CL-count-updated"])]
index_cvs_only = int(df_models_cvs.index[0])
print_results_for_index(index_cvs_only)

df_results.to_csv("/home/mwynen/data/MSDT/test_set_prodromal_metrics.csv", index=False)


