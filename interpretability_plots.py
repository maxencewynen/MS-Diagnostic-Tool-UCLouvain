# explore interpretability of models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
import joblib
import xgboost as xgb
import shap
from utils import *
import os
from os.path import join as pjoin


def get_features_from_model(model):
    if isinstance(model, xgb.XGBClassifier):
        return model.get_booster().feature_names
    else:
        return model.feature_names_in_


def prettify_feature_names(shap_values):
    pretty_names_for_plot = {
        "Thompson": "DIS",
        "number_PRL": "Number of PRL",
        "PRL1": "Presence of PRL",
        "% perivenular les": "% CVS",
        "Select3*-v2NA_0.0": "Select 3 (CVS) = ✓",
        "Select3*-v2NA_1.0": "Select 3 (CVS) = ✖",
        "Select3*-v2NA_NA": "Select 3 (CVS) = NA",
        "Select6*-v2NA_0.0": "Select 6 (CVS) = ✓",
        "Select6*-v2NA_1.0": "Select 6 (CVS) = ✖",
        "Select6*-v2NA_NA": "Select 6 (CVS) = NA",
        "CL-count-updated": "Number of CL",
        "CL1": "Presence of CL",
        "age": "Age",
        "sex": "Sex",
    }

    shap_values.feature_names = [pretty_names_for_plot.get(feature, feature) for feature in shap_values.feature_names]
    return shap_values


def get_explainer_and_shap_values(model, nsamples=50):
    features = get_features_from_model(model)
    X_tr, y_tr = get_full_dataset(TRAINING_DATA_PATH, features=features, dropna=False, return_X_y=True, prepare=True)
    X_te, y_te = get_full_dataset(TEST_DATA_PATH, features=features, dropna=False, return_X_y=True, prepare=True)

    if isinstance(model, xgb.XGBClassifier):
        explainer = shap.Explainer(model)
        shap_values = explainer(X_te)
    elif isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, masker=X_tr)
        shap_values = explainer(X_te)
        shap_values.values = shap_values.values.astype(np.float32)
    elif isinstance(model, RandomForestClassifier):
        # explainer = shap.TreeExplainer(model)
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_tr, nsamples))
        shap_values = explainer(X_te)
        shap_values.values = shap_values.values[:,:,1]
        shap_values.base_values = shap_values.base_values[:,1]
    elif isinstance(model, SVC):
        explainer = shap.KernelExplainer(model.predict, shap.sample(X_tr, nsamples))
        shap_values = explainer(X_te)
    else:
        raise ValueError("Model not supported")

    return explainer, shap_values


def save_explainer_and_shap_values(data_path, results_path):
    df_results = pd.read_csv(results_path)
    df_results = df_results[(df_results["f1"] >= 0.947) & (df_results["balanced_accuracy"] >= 0.936)]

    shap_values_paths = []
    explainer_paths = []

    for _, row in df_results.iterrows():
        model = joblib.load(row["model_file"])
        explainer, shap_values = get_explainer_and_shap_values(model, data_path)
        shap_values = prettify_feature_names(shap_values)

        output_file_shap = row['model_file'].replace('.pkl','_shap_values.pkl')
        joblib.dump(shap_values, output_file_shap)

        output_file_explainer = row['model_file'].replace('.pkl','_explainer.pkl')
        joblib.dump(explainer, output_file_explainer)

        shap_values_paths.append(output_file_shap)
        explainer_paths.append(output_file_explainer)

    df_results['shap_values_file'] = shap_values_paths
    df_results['explainer_file'] = explainer_paths

    return df_results


def get_number_of_models_per_most_important_feature(results_path):
    n_models_per_most_important_feature = {}
    n_models_where_feature_is_used_as_input = {}
    df_results = pd.read_csv(results_path)
    df_results = df_results[(df_results["f1"] >= 0.947) & (df_results["balanced_accuracy"] >= 0.936)]
    for _, row in df_results.iterrows():
        print(row["model_name"])
        model = joblib.load(row["model_file"])

        explainer, shap_values = get_explainer_and_shap_values(model, data_path)
        shap_values = prettify_feature_names(shap_values)

        # get absolute shap values per feature
        abs_shap_value_per_feature = np.abs(shap_values.values).mean(0)
        abs_shap_value_per_feature = pd.DataFrame(abs_shap_value_per_feature, index=shap_values.feature_names,
                                                  columns=["SHAP"])
        most_important_feature = abs_shap_value_per_feature.idxmax().values[0]

        if most_important_feature not in n_models_per_most_important_feature:
            n_models_per_most_important_feature[most_important_feature] = 1
        else:
            n_models_per_most_important_feature[most_important_feature] += 1

        for feature in shap_values.feature_names:
            if feature not in n_models_where_feature_is_used_as_input:
                n_models_where_feature_is_used_as_input[feature] = 1
            else:
                n_models_where_feature_is_used_as_input[feature] += 1

    percentages = {}
    for feature in n_models_where_feature_is_used_as_input:
        percentages[feature] = round((n_models_per_most_important_feature.get(feature, 0) / n_models_where_feature_is_used_as_input[feature]) * 100, 2)

    return percentages


def make_model_stats_per_feature(results_path):
    pretty_names_for_plot = {
        "Filippi": "DIS (Filippi)",
        "Thompson": "DIS (McDonald)",
        "number_PRL": "Number of PRL",
        "PRL1": "Presence of PRL",
        "% perivenular les": "% CVS",
        "Select3*-v2NA_0.0": "Select 3 (CVS) = ✓",
        "Select3*-v2NA_1.0": "Select 3 (CVS) = ✖",
        "Select3*-v2NA_NA": "Select 3 (CVS) = NA",
        "Select3*-v2NA": "Select 3 (CVS)",
        "Select6*-v2NA_0.0": "Select 6 (CVS) = ✓",
        "Select6*-v2NA_1.0": "Select 6 (CVS) = ✖",
        "Select6*-v2NA_NA": "Select 6 (CVS) = NA",
        "Select6*-v2NA": "Select 6 (CVS)",
        "CL-count-updated": "Number of CL",
        "CL1": "Presence of CL",
        "OCB_presence": "Presence of OCB",
        "age": "Age",
        "sex": "Sex",
    }

    old_name_to_prety_name = {v: k for k, v in pretty_names_for_plot.items()}

    from utils import ALL_FEATURES
    model_stats = {pf: {1: 0, 2: 0, 'total_count': 0} for of, pf in pretty_names_for_plot.items()
                   if (not of.endswith("_NA") and not of.endswith("_0.0") and not of.endswith("_1.0"))}
    df_results = pd.read_csv(results_path)
    df_results = df_results[(df_results["f1"] >= 0.947) & (df_results["balanced_accuracy"] >= 0.936)]
    for _, row in df_results.iterrows():
        model = joblib.load(row["model_file"])

        explainer, shap_values = get_explainer_and_shap_values(model, data_path)
        shap_values = prettify_feature_names(shap_values)

        # get absolute shap values per feature
        abs_shap_value_per_feature = np.abs(shap_values.values).mean(0)
        abs_shap_value_per_feature = pd.DataFrame(abs_shap_value_per_feature, index=shap_values.feature_names,
                                                  columns=["SHAP"])

        # Sort by importance
        abs_shap_value_per_feature = abs_shap_value_per_feature.sort_values(by="SHAP", ascending=False)
        for i, feature in enumerate(abs_shap_value_per_feature.index):
            # print(feature)
            if feature.endswith("_NA") or feature.endswith("_0.0") or feature.endswith("_1.0"):
                feature = feature[:-4]
            if feature.endswith(" = ✓") or feature.endswith(" = ✖") or feature.endswith(" = NA"):
                feature = feature[:-4]
            feature = feature.strip()
            # print(feature)
            # feature = old_name_to_prety_name.get(feature, feature)
            # print(feature)
            if i == 0:
                model_stats[feature][1] += 1
            elif i == 1:
                model_stats[feature][2] += 1
            model_stats[feature]['total_count'] += 1

    model_stats["Select 3 (CVS)"]["total_count"] //= 3
    model_stats["Select 6 (CVS)"]["total_count"] //= 3
    # model_stats = pd.DataFrame(model_stats).T
    print(model_stats)
    return model_stats


def make_plot_for_model(X_tr, X_te, model_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = joblib.load(model_path)
    features = get_features_from_model(model)
    X_tr = X_tr[features]
    X_te = X_te[features]

    if isinstance(model, xgb.XGBClassifier):
        explainer = shap.Explainer(model)
        shap_values = explainer(X_te)
    elif isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, masker=X_tr)
        shap_values = explainer(X_te)
        shap_values.values = shap_values.values.astype(np.float32)
    elif isinstance(model, RandomForestClassifier):
        # explainer = shap.TreeExplainer(model)
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_tr, 300))
        shap_values = explainer(X_te)
        shap_values.values = shap_values.values[:, :, 1]
        shap_values.base_values = shap_values.base_values[:, 1]
    elif isinstance(model, SVC):
        explainer = shap.KernelExplainer(model.predict, shap.sample(X_tr, 300))
        shap_values = explainer(X_te)
    else:
        return
        # raise ValueError(f"Model {model} not supported")

    shap_values = prettify_feature_names(shap_values)

    # get absolute shap values per feature
    abs_shap_value_per_feature = np.abs(shap_values.values).mean(0)
    abs_shap_value_per_feature = pd.DataFrame(abs_shap_value_per_feature, index=shap_values.feature_names,
                                              columns=["SHAP"])

    # Beeswarm plot
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, os.path.basename(model_path).replace(".pkl", "_beeswarm_plot.png")))
    plt.close()

    # Summary plot
    ax = shap.summary_plot(shap_values, X_te, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, os.path.basename(model_path).replace(".pkl", "_summary_plot.png")))
    plt.close()


if __name__ == "__main__":
    X_tr, y_tr = get_full_dataset(TRAINING_DATA_PATH, dropna=False, return_X_y=True, prepare=True)
    X_te, y_te = get_full_dataset(TEST_DATA_PATH, dropna=False, return_X_y=True, prepare=True)

    all_models = pd.read_csv(ALL_MODELS_FOR_ONLINE_TOOL_PATH)

    for i, row in all_models.iterrows():
        make_plot_for_model(X_tr, X_te, row["model_path"], PLOTS_OUTPUT_PATH)