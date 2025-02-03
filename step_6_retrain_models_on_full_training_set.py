import numpy as np
from utils import *
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
import warnings
import os
import pickle
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report, f1_score
warnings.filterwarnings("ignore", category=FutureWarning)


def sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    return sensitivity


def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity


def precision(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp)
    return precision

def negative_predictive_value(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    negative_predictive_value = tn / (tn + fn)
    return negative_predictive_value


def f1_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


# df_models = pd.read_csv(MODELS_OUTPERFORMING_MCDONALD_PATH)
df_models = pd.read_csv(ALL_MODELS_REGARDLESS_OF_PERFORMANCE_PATH)

X, y = get_full_dataset(TRAINING_DATA_PATH, dropna=False, return_X_y=True, prepare=True)

all_models_for_online_tool = []

for i, row in df_models.iterrows():
    print(f"Progress: {i + 1}/{len(df_models)}")
    model_name = row["model"]
    threshold = row["threshold"]
    features = [f for f in row.index if f in ALL_FEATURES and row[f]]
    print(features)
    c = [f for f in features if f in X.columns and f not in ('Select3*-v2NA', 'Select6*-v2NA')]
    if "Select3*-v2NA" in features:
        c += ["Select3*-v2NA_0.0", "Select3*-v2NA_1.0", "Select3*-v2NA_NA"]
    if "Select6*-v2NA" in features:
        c += ["Select6*-v2NA_0.0", "Select6*-v2NA_1.0", "Select6*-v2NA_NA"]
    _X = X[c].copy()
    # Remove rows with missing values in columns from comb
    _X = _X.dropna()
    _y = y[_X.index]

    model = model_name + '()' if not model_name.endswith('()') and \
                                 not model_name.endswith('(probability=True)') else model_name
    model = eval(model)


    # Save the model
    pretty_features = [f.replace(" ", "-") for f in features]
    model_path = os.path.join(MODELS_OUTPUT_DIR, f"{model_name}_{'_'.join(pretty_features)}.pkl")

    # REMOVE IF YOU WANT TO RE-TRAIN MODELS
    if not os.path.exists(model_path):
        model.fit(_X, _y)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

    row['model_path'] = model_path

    # metrics
    y_pred_proba = model.predict_proba(_X)[:, 1]
    y_pred = y_pred_proba > threshold

    expected_balanced_accuracy = row['avg_balanced_accuracy']
    row['expected_balanced_accuracy'] = expected_balanced_accuracy

    row['tr_balanced_accuracy'] = balanced_accuracy_score(_y, y_pred)
    row['tr_sensitivity'] = sensitivity(_y, y_pred)
    row['tr_specificity'] = specificity(_y, y_pred)
    row['tr_precision'] = precision(_y, y_pred)
    row['tr_f1'] = f1_score(_y, y_pred)
    row['tr_negative_predictive_value'] = negative_predictive_value(_y, y_pred)

    all_models_for_online_tool.append(row)

all_models_for_online_tool = pd.DataFrame(all_models_for_online_tool)
all_models_for_online_tool.to_csv(ALL_MODELS_FOR_ONLINE_TOOL_PATH, index=False)





