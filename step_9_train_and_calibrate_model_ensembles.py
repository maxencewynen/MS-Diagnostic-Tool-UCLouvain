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
from sklearn.calibration import CalibratedClassifierCV
import warnings
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report, f1_score
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.utils.validation import _num_samples, check_is_fitted


def predict_proba_and_std(self, X):
    """Calibrated probabilities of classification.

    This function returns calibrated probabilities of classification
    according to each class on an array of test vectors X.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples, as accepted by `estimator.predict_proba`.

    Returns
    -------
    C : ndarray of shape (n_samples, n_classes)
        The predicted probas.
    """
    check_is_fitted(self)
    # Compute the arithmetic mean of the predictions of the calibrated
    # classifiers
    mean_proba = np.zeros((_num_samples(X), len(self.classes_)))
    all_proba = np.zeros((_num_samples(X), len(self.classes_), len(self.calibrated_classifiers_)))
    for i, calibrated_classifier in enumerate(self.calibrated_classifiers_):
        proba = calibrated_classifier.predict_proba(X)
        mean_proba += proba
        all_proba[:, :, i] = proba

    mean_proba /= len(self.calibrated_classifiers_)
    std_proba = np.std(all_proba, axis=2)

    return mean_proba, std_proba


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
# We will use the test set to calibrate the models
X_test_verona, y_test_verona = get_full_dataset(TEST_DATA_PATH, dropna=False, return_X_y=True, prepare=True)
X_test_verona = X_test_verona.dropna(subset=["% perivenular les"])
y_test_verona = y_test_verona[X_test_verona.index]

X_test_prodromal, y_test_prodromal = get_full_dataset(PRODROMAL_TEST_DATA_PATH, dropna=False, return_X_y=True, prepare=True)
X_test_prodromal = X_test_prodromal.dropna(subset=["% perivenular les"])
y_test_prodromal = y_test_prodromal[X_test_prodromal.index]

X_test = pd.concat([X_test_verona, X_test_prodromal]).astype(np.float64)
y_test = pd.concat([y_test_verona, y_test_prodromal]).astype(int)
# Drop nan


all_models_for_online_tool = []

for i, row in df_models.iterrows():
    print(f"Progress: {i + 1}/{len(df_models)}")
    orig_model_name = row["model"]
    features = [f for f in row.index if f in ALL_FEATURES and row[f]]
    print(orig_model_name, features)
    c = [f for f in features if f in X.columns and f not in ('Select3*-v2NA', 'Select6*-v2NA')]
    if "Select3*-v2NA" in features:
        c += ["Select3*-v2NA_0.0", "Select3*-v2NA_1.0", "Select3*-v2NA_NA"]
    if "Select6*-v2NA" in features:
        c += ["Select6*-v2NA_0.0", "Select6*-v2NA_1.0", "Select6*-v2NA_NA"]
    _X = X[c].copy()
    # Remove rows with missing values in columns from comb
    _X = _X.dropna()
    _y = y[_X.index]

    _X_test = X_test[c].copy()
    # Remove rows with missing values in columns from comb
    _X_test = _X_test.dropna()
    _y_test = y_test[_X_test.index]

    model_name = orig_model_name + '()' if not orig_model_name.endswith('()') and \
                                   not orig_model_name.endswith('(probability=True)') else orig_model_name
    model = eval(model_name)

    pretty_features = [f.replace(" ", "-") for f in features]
    model_path = os.path.join(FINAL_MODELS_OUTPUT_DIR, f"{orig_model_name}_{'_'.join(pretty_features)}.pkl")

    # calibrate model output on test data
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5).fit(_X, _y)
    calibrated_model.predict_proba_and_std = predict_proba_and_std

    with open(model_path, 'wb') as f:
        pickle.dump(calibrated_model, f)

    row['model_path'] = model_path

    # metrics
    y_pred_proba = calibrated_model.predict_proba(_X)[:, 1]
    y_pred = y_pred_proba > 0.5  # since the model is calibrated, we can use 0.5 as threshold

    expected_balanced_accuracy = row['avg_balanced_accuracy']
    row['expected_balanced_accuracy'] = expected_balanced_accuracy

    row['tr_balanced_accuracy'] = balanced_accuracy_score(_y, y_pred)
    row['tr_sensitivity'] = sensitivity(_y, y_pred)
    row['tr_specificity'] = specificity(_y, y_pred)
    row['tr_precision'] = precision(_y, y_pred)
    row['tr_f1'] = f1_score(_y, y_pred)
    row['tr_negative_predictive_value'] = negative_predictive_value(_y, y_pred)

    y_test_pred_proba, y_test_pred_std = predict_proba_and_std(calibrated_model, _X_test)
    y_test_pred_proba = y_test_pred_proba[:, 1]
    y_test_pred_std = y_test_pred_std[:, 1]
    y_test_pred = y_test_pred_proba > 0.5  # since the model is calibrated, we can use 0.5 as threshold
    # ax = sns.histplot(y_test_pred_proba[len(y_test_verona):], binwidth=0.025)
    # ax.set_title(f"{orig_model_name} - {'_'.join(pretty_features)}")
    # plt.show()
    row['te_balanced_accuracy'] = balanced_accuracy_score(_y_test, y_test_pred)
    row['te_sensitivity'] = sensitivity(_y_test, y_test_pred)
    row['te_specificity'] = specificity(_y_test, y_test_pred)
    row['te_precision'] = precision(_y_test, y_test_pred)
    row['te_f1'] = f1_score(_y_test, y_test_pred)
    row['te_negative_predictive_value'] = negative_predictive_value(_y_test, y_test_pred)

    row['te_verona_balanced_accuracy'] = balanced_accuracy_score(_y_test[:len(y_test_verona)], y_test_pred[:len(y_test_verona)])
    row['te_verona_sensitivity'] = sensitivity(_y_test[:len(y_test_verona)], y_test_pred[:len(y_test_verona)])
    row['te_verona_specificity'] = specificity(_y_test[:len(y_test_verona)], y_test_pred[:len(y_test_verona)])
    row['te_verona_verona_specificity'] = specificity(_y_test[:len(y_test_verona)], y_test_pred[:len(y_test_verona)])
    row['te_verona_precision'] = precision(_y_test[:len(y_test_verona)], y_test_pred[:len(y_test_verona)])
    row['te_verona_f1'] = f1_score(_y_test[:len(y_test_verona)], y_test_pred[:len(y_test_verona)])
    row['te_verona_negative_predictive_value'] = negative_predictive_value(_y_test[:len(y_test_verona)], y_test_pred[:len(y_test_verona)])

    row['te_prodromal_balanced_accuracy'] = balanced_accuracy_score(_y_test[len(y_test_verona):], y_test_pred[len(y_test_verona):])
    row['te_prodromal_sensitivity'] = sensitivity(_y_test[len(y_test_verona):], y_test_pred[len(y_test_verona):])
    row['te_prodromal_specificity'] = specificity(_y_test[len(y_test_verona):], y_test_pred[len(y_test_verona):])
    row['te_prodromal_precision'] = precision(_y_test[len(y_test_verona):], y_test_pred[len(y_test_verona):])
    row['te_prodromal_f1'] = f1_score(_y_test[len(y_test_verona):], y_test_pred[len(y_test_verona):])
    row['te_prodromal_negative_predictive_value'] = negative_predictive_value(_y_test[len(y_test_verona):], y_test_pred[len(y_test_verona):])

    print("Balanced accuracy Train set: ", round(balanced_accuracy_score(_y, y_pred)*100,1))
    print("Balanced accuracy Test set: ", round(balanced_accuracy_score(_y_test, y_test_pred)*100,1))
    print("Balanced accuracy Verona: ", round(balanced_accuracy_score(_y_test[:len(y_test_verona)], y_test_pred[:len(y_test_verona)])*100,1))
    print("Balanced accuracy Prodromal: ", round(balanced_accuracy_score(_y_test[len(y_test_verona):], y_test_pred[len(y_test_verona):])*100,1))

    all_models_for_online_tool.append(row)

    # # Train 10 models
    # for random_seed in tqdm(range(10)):
    #     if orig_model_name.endswith('()'):
    #         model_name = orig_model_name.replace('()', f'(random_state={random_seed})')
    #     elif orig_model_name.endswith('(probability=True)'):
    #         model_name = orig_model_name.replace('(probability=True)', f'(probability=True, random_state={random_seed})')
    #     else:
    #         model_name = orig_model_name + f'(random_state={random_seed})'
    #
    #     model = eval(model_name)
    #     pretty_features = [f.replace(" ", "-") for f in features]
    #     model_path = os.path.join(FINAL_MODELS_OUTPUT_DIR, f"{orig_model_name}_{'_'.join(pretty_features)}_seed-{random_seed}.pkl")
    #
    #     model.fit(_X, _y)
    #
    #     # calibrate model output on test data
    #     calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit').fit(_X_test, _y_test)
    #
    #     with open(model_path, 'wb') as f:
    #         pickle.dump(calibrated_model, f)
    #
    #     row['model_path'] = model_path
    #
    #     # metrics
    #     y_pred_proba = calibrated_model.predict_proba(_X)[:, 1]
    #     y_pred = y_pred_proba > 0.5  # since the model is calibrated, we can use 0.5 as threshold
    #
    #     expected_balanced_accuracy = row['avg_balanced_accuracy']
    #     row['expected_balanced_accuracy'] = expected_balanced_accuracy
    #
    #     row['tr_balanced_accuracy'] = balanced_accuracy_score(_y, y_pred)
    #     row['tr_sensitivity'] = sensitivity(_y, y_pred)
    #     row['tr_specificity'] = specificity(_y, y_pred)
    #     row['tr_precision'] = precision(_y, y_pred)
    #     row['tr_f1'] = f1_score(_y, y_pred)
    #     row['tr_negative_predictive_value'] = negative_predictive_value(_y, y_pred)
    #
    #     # row['test_balanced_accuracy'] = balanced_accuracy_score(y_test, calibrated_model.predict(X_test))
    #     # row['test_sensitivity'] = sensitivity(_y, y_pred)
    #     # row['test_specificity'] = specificity(_y, y_pred)
    #     # row['test_precision'] = precision(_y, y_pred)
    #     # row['test_f1'] = f1_score(_y, y_pred)
    #     # row['test_negative_predictive_value'] = negative_predictive_value(_y, y_pred)
    #
    #     all_models_for_online_tool.append(row)

all_models_for_online_tool = pd.DataFrame(all_models_for_online_tool)
all_models_for_online_tool.to_csv(ALL_CALIBRATED_ENSEMBLE_MODELS_FOR_ONLINE_TOOL_PATH, index=False)





