import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

ALL_FEATURES = [
    'Thompson',
    '% perivenular les',
    'Select3*-v2NA',
    'Select6*-v2NA',
    'CL-count-updated',
    'CL1',
    'PRL1',
    'number_PRL'
]

ALL_FEATURES_FOR_ML_INPUT = [
    'Thompson',
    '% perivenular les',
    'Select3*-v2NA_NA',
    'Select3*-v2NA_0.0',
    'Select3*-v2NA_1.0',
    'Select6*-v2NA_NA',
    'Select6*-v2NA_0.0',
    'Select6*-v2NA_1.0',
    'CL-count-updated',
    'CL1',
    'PRL1',
    'number_PRL'
]

feature_groups = ["PRL", "CVS", "CL"]
group_values = {
    # "DIS": ["Filippi", "Thompson"],
    "PRL": ["number_PRL", "PRL1"],
    "CVS": ["% perivenular les", "Select3*-v2NA", "Select6*-v2NA"],
    "CL": ["CL-count-updated", "CL1"],
}

simplified_features = ["Select3*-v2NA", "Select6*-v2NA", "PRL1", "CL1"]
full_features = ["% perivenular les", "number_PRL", "CL-count-updated"]

simplified_to_full = {
    "Select3*-v2NA": "% perivenular les",
    "Select6*-v2NA": "% perivenular les",
    "PRL1": "number_PRL",
    "CL1": "CL-count-updated"
}

TRAINING_DATA_PATH = "/home/mwynen/data/MSDT/training_set.xlsx"
TEST_DATA_PATH = "/home/mwynen/data/MSDT/test_set.xlsx"
BEST_ALGORITHM_PER_COMBINATION_OF_FEATURES_PATH = "/home/mwynen/data/MSDT/results/best_algorithm_per_combination_of_features.csv"
MODELS_OUTPERFORMING_MCDONALD_PATH = "/home/mwynen/data/MSDT/results/models_outperforming_mcdonald.csv"
ALL_MODELS_REGARDLESS_OF_PERFORMANCE_PATH = "/home/mwynen/data/MSDT/results/all_models.csv"
MODELS_OUTPUT_DIR = "/home/mwynen/data/MSDT/results/models/"
FINAL_MODELS_OUTPUT_DIR = "/home/mwynen/data/MSDT/results/final_models/"
ALL_MODELS_FOR_ONLINE_TOOL_PATH = "/home/mwynen/data/MSDT/results/all_models_for_online_tool.csv"
ALL_CALIBRATED_ENSEMBLE_MODELS_FOR_ONLINE_TOOL_PATH = "/home/mwynen/data/MSDT/results/all_calibrated_ensemble_models_for_online_tool.csv"
TEST_PREDICTIONS_PATH = "/home/mwynen/data/MSDT/results/test_predictions.csv"
RESULTS_TXT = "/home/mwynen/data/MSDT/results/results.txt"
PLOTS_OUTPUT_PATH = "/home/mwynen/data/MSDT/results/plots/"
PRODROMAL_TEST_DATA_PATH = "/home/mwynen/data/MSDT/test_set_prodromal.xlsx"
PRODROMAL_TEST_PREDICTIONS_PATH = "/home/mwynen/data/MSDT/test_set_prodromal_predictions.csv"

USE_PIPELINE = False

RANDOM_SEED = 42


def pprint(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, file=open(RESULTS_TXT, "a"), **kwargs)


def prepare_data(*args):
    """
    Prepare the data for training.
    :param args: (tuple) Tuple containing the input data and the target variable.
    :return: X, y
        X: (pandas.DataFrame) Training dataset with only the input features, removing the rows with missing values
            for those features.
        y: (pandas.Series) Training dataset with only the target variable ("diagnosis").
    """
    if len(args) == 2:
        X, y = args
        return _prepare_data_for_training(X, y)
    elif len(args) == 1:
        return _prepare_data_df(args[0])


def _prepare_data_df(df):
    """
    Prepare the data for training.
    :param df: pd.DataFrame containing the data.
    :return: df: pd.DataFrame containing the data with the following modifications:
        - Replace missing values in Split3* and Split6* columns with new category and one-hot encode
        - Replace following columns by booleans: "sex", "Thompson", "Filippi", "diagnosis"
        - Set subject column as id
    """
    # Replace missing values in Split3* and Split6* columns with new category and one-hot encode
    if "Select3*-v2NA" in df.columns:
        df["Select3*-v2NA"] = df["Select3*-v2NA"].fillna("NA")
    if "Select6*-v2NA" in df.columns:
        df["Select6*-v2NA"] = df["Select6*-v2NA"].fillna("NA")
    columns = [x for x in ["Select3*-v2NA", "Select6*-v2NA"] if x in df.columns]
    if len(columns) > 0:
        df = pd.get_dummies(df, columns=columns)

    # Replace following columns by booleans
    if "sex" in df.columns:
        df["sex"] = df["sex"].replace({1: True, 0: False})
    if "Thompson" in df.columns:
        df["Thompson"] = df["Thompson"].replace({1: True, 0: False})
    if "Filippi" in df.columns:
        df["Filippi"] = df["Filippi"].replace({1: True, 0: False})

    df["diagnosis"] = df["diagnosis"].replace({1: True, 0: False})

    if "subject" in df.columns:
        # set subject column as id
        df = df.set_index("subject")

    # Drop columns that are not in the ALL_FEATURES list
    df = df[ALL_FEATURES_FOR_ML_INPUT + ["diagnosis"]]

    return df


def _prepare_data_for_training(X, y):
    """
    Prepare the data for training.
    :param X: pd.DataFrame containing the input data.
    :param y: pd.Series containing the target variable.
    :return: df: pd.DataFrame containing the data with the following modifications:
        - Replace missing values in Select3* and Select* columns with new category and one-hot encode
        - Replace following columns by booleans: "sex", "Thompson", "Filippi", "diagnosis"
        - Set subject column as id
    """
    df = pd.concat([X, y], axis=1)
    df = _prepare_data_df(df)
    y = df["diagnosis"]
    X = df.drop(columns=["diagnosis"])

    return X, y


def get_full_dataset(data_path, features=None, dropna=True, prepare=True, return_X_y=False):
    df = pd.read_excel(data_path)
    if prepare:
        df = _prepare_data_df(df)
        # Replace Select3*-v2NA and Select6*-v2NA columns by Select3*-v2NA_NA, Select3*-v2NA_0.0, Select3*-v2NA_1.0,
        # Select6*-v2NA_NA, Select6*-v2NA_0.0, Select6*-v2NA_1.0 columns
        if features is not None:
            other_feats = [f for f in features if f not in ["Select3*-v2NA", "Select6*-v2NA"]]
            feats_to_be_added = []
            if 'Select3*-v2NA' in features:
                feats_to_be_added += ["Select3*-v2NA_NA", "Select3*-v2NA_0.0", "Select3*-v2NA_1.0"]
            if 'Select6*-v2NA' in features:
                feats_to_be_added += ["Select6*-v2NA_NA", "Select6*-v2NA_0.0", "Select6*-v2NA_1.0"]
            features = other_feats + feats_to_be_added

    if features is not None:
        df = df[features + ["diagnosis"]]

    if dropna:
        df = df.dropna()

    if return_X_y:
        y = df["diagnosis"]
        X = df.drop(columns=["diagnosis"])
        return X, y
    return df


def combined_ftest_5x2cv_balanced_accuracy_with_threshold(estimator1, estimator2, threshold1, threshold2, comb1, comb2,
                                                          X, y, random_seed=None):
    """
    ADAPTED FROM mlxtend.evaluate.combined_ftest_5x2cv
    Implements the 5x2cv combined F test proposed
    by Alpaydin 1999,
    to compare the performance of two models.

    Parameters
    ----------
    estimator1 : scikit-learn classifier or regressor

    estimator2 : scikit-learn classifier or regressor

    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape = [n_samples]
        Target values.

    scoring : str, callable, or None (default: None)
        If None (default), uses 'accuracy' for sklearn classifiers
        and 'r2' for sklearn regressors.
        If str, uses a sklearn scoring metric string identifier, for example
        {accuracy, f1, precision, recall, roc_auc} for classifiers,
        {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',
        'median_absolute_error', 'r2'} for regressors.
        If a callable object or function is provided, it has to be conform with
        sklearn's signature ``scorer(estimator, X, y)``; see
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        for more information.

    random_seed : int or None (default: None)
        Random seed for creating the test/train splits.

    Returns
    ----------
    f : float
        The F-statistic

    pvalue : float
        Two-tailed p-value.
        If the chosen significance level is larger
        than the p-value, we reject the null hypothesis
        and accept that there are significant differences
        in the two compared models.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/evaluate/combined_ftest_5x2cv/

    """
    rng = np.random.RandomState(random_seed)

    variances = []
    differences = []

    def score_diff(X_1, X_2, y_1, y_2):
        X_2 = X_2[list(set(comb1).union(set(comb2)))].dropna()
        y_2 = y_2[X_2.index]

        estimator1.fit(X_1[comb1], y_1)
        estimator2.fit(X_1[comb2], y_1)

        y2_pred_proba_estimator1 = estimator1.predict_proba(X_2[comb1])
        try:
            y2_pred_estimator1 = (y2_pred_proba_estimator1[:, 1] >= threshold1).astype(int)
        except KeyError:
            y2_pred_estimator1 = (y2_pred_proba_estimator1 >= threshold1).astype(int)

        y2_pred_proba_estimator2 = estimator2.predict_proba(X_2[comb2])
        try:
            y2_pred_estimator2 = (y2_pred_proba_estimator2[:, 1] >= threshold2).astype(int)
        except KeyError:
            y2_pred_estimator2 = (y2_pred_proba_estimator2 >= threshold2).astype(int)

        est1_score = balanced_accuracy_score(y_2, y2_pred_estimator1)
        est2_score = balanced_accuracy_score(y_2, y2_pred_estimator2)
        score_diff = est1_score - est2_score
        return score_diff, est1_score, est2_score

    all_score_diffs = []
    all_scores_est1 = []
    all_scores_est2 = []

    for i in range(5):
        randint = rng.randint(low=0, high=32767)
        X_1, X_2, y_1, y_2 = train_test_split(X, y, test_size=0.5, random_state=randint)

        score_diff_1, est1_score_1, est2_score_1 = score_diff(X_1, X_2, y_1, y_2)
        score_diff_2, est1_score_2, est2_score_2 = score_diff(X_2, X_1, y_2, y_1)

        all_score_diffs.extend([score_diff_1, score_diff_2])
        all_scores_est1.extend([est1_score_1, est1_score_2])
        all_scores_est2.extend([est2_score_1, est2_score_2])

        score_mean = (score_diff_1 + score_diff_2) / 2.0
        score_var = (score_diff_1 - score_mean) ** 2 + (score_diff_2 - score_mean) ** 2

        differences.extend([score_diff_1**2, score_diff_2**2])
        variances.append(score_var)

    numerator = sum(differences)
    denominator = 2 * (sum(variances))
    f_stat = numerator / (denominator + 1e-8)

    p_value = scipy.stats.f.sf(f_stat, 10, 5)

    mean_diff = np.mean(all_score_diffs)
    mean_score_est1 = np.mean(all_scores_est1)
    mean_score_est2 = np.mean(all_scores_est2)

    lower_ci = np.percentile(all_score_diffs, 100 * 0.05 / 2)
    upper_ci = np.percentile(all_score_diffs, 100 * (1 - 0.05 / 2))

    res = {
        "f_stat": float(f_stat),
        "p_value": float(p_value),
        "mean_diff": mean_diff,
        "mean_score_est1": mean_score_est1,
        "mean_score_est2": mean_score_est2,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
    }

    return res


class BaselineMcDonald:
    """
    Rule based classifier based on the McDonald criteria.
    """
    def __init__(self):
        self._estimator_type = "classifier"
        self.classes_ = [False, True]

    def fit(self, X, y):
        pass

    def predict(self, X):
        # return (X["OCB_presence"].astype(bool) & X["Thompson"].astype(bool)).astype(int)
        return X["Thompson"].astype(bool).astype(int)

    def predict_proba(self, X):
        return self.predict(X).astype(float)


def compare_model_with_other_models(X,y, models_df, index=0, index2=None, pipeline=False):
    # get row at index
    model_of_interest_row = models_df.iloc[index]
    model_of_interest = model_of_interest_row["model"]
    model_of_interest_threshold = model_of_interest_row["threshold"]
    model_of_interest_features = [f for f in model_of_interest_row.index if f not in ("model", "threshold")
                                    and model_of_interest_row[f] and f in ALL_FEATURES]
    if model_of_interest == 'SVC()':
        model_of_interest = 'SVC(probability=True)'
    if pipeline:
        model_of_interest = Pipeline([("scaler", StandardScaler()), ("model", eval(model_of_interest))])
    else:
        model_of_interest = eval(model_of_interest)
    results = []

    X, y = prepare_data(X, y)
    c = [f for f in model_of_interest_features if f in X.columns and f not in ('Select3*-v2NA', 'Select6*-v2NA')]
    if "Select3*-v2NA" in model_of_interest_features:
        c += ["Select3*-v2NA_0.0", "Select3*-v2NA_1.0", "Select3*-v2NA_NA"]
    if "Select6*-v2NA" in model_of_interest_features:
        c += ["Select6*-v2NA_0.0", "Select6*-v2NA_1.0", "Select6*-v2NA_NA"]
    model_of_interest_features = comb = c
    _X = X[comb].copy()
    # Remove rows with missing values in columns from comb
    _X = _X.dropna()
    y = y[_X.index]
    # Only keep the with index in _X.index
    X = X.loc[_X.index]
    random_seed = RANDOM_SEED

    j = -1
    for i, row in models_df.iterrows():
        j+=1
        if index2 is None:
            if j == index:
                print(f"Progress: {j + 1}/{len(models_df)}: Skipping model of interest. {model_of_interest_features}")
                continue
            print(f"Progress: {j + 1}/{len(models_df)}", end=" ")
        elif i != index2:
            continue

        model = row["model"]
        threshold = row["threshold"]
        features = [f for f in row.index if f not in ("model", "threshold") and row[f] and f in ALL_FEATURES]
        c = [f for f in features if f in X.columns and f not in ('Select3*-v2NA', 'Select6*-v2NA')]
        if "Select3*-v2NA" in features:
            c += ["Select3*-v2NA_0.0", "Select3*-v2NA_1.0", "Select3*-v2NA_NA"]
        if "Select6*-v2NA" in features:
            c += ["Select6*-v2NA_0.0", "Select6*-v2NA_1.0", "Select6*-v2NA_NA"]
        features = c
        print(features)

        if model in ['SVC()', 'SVC', 'SVC(probability=True)']:
            model = 'SVC(probability=True)'
        else:
            model = model + '()' if not model.endswith('()') else model
        if pipeline:
            model = Pipeline([("scaler", StandardScaler()), ("model", eval(model))])
        else:
            model = eval(model)

        _X = X[features].copy()
        # Remove rows with missing values in columns from comb
        _X = _X.dropna()
        _y = y[_X.index]
        # Only keep the with index in _X.index
        _X = X.loc[_X.index]

        res = combined_ftest_5x2cv_balanced_accuracy_with_threshold(estimator1=model_of_interest,
                                                                    estimator2=model,
                                                                    threshold1=model_of_interest_threshold,
                                                                    threshold2=threshold,
                                                                    comb1=model_of_interest_features,
                                                                    comb2=features,
                                                                    random_seed=random_seed,
                                                                    X=_X, y=_y)

        if 'XGB' in str(model):
            model = 'XGBClassifier()'

        results.append({
            "model": model,
            "threshold": threshold,
            "mean_score_model": res["mean_score_est1"],
            "mean_score_model2": res["mean_score_est2"],
            "pval": res["p_value"],
            "f_stat": res["f_stat"],
            "mean_diff": res["mean_diff"],
        })

    return results


