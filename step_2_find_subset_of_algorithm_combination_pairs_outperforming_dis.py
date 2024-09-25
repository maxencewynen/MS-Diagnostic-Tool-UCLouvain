from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import numpy as np
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
import os
import warnings
np.random.seed(RANDOM_SEED)
warnings.filterwarnings("ignore", category=FutureWarning)

def compare_model_with_baseline(model, threshold, comb, X, y):
    X, y = prepare_data(X, y)
    c = [f for f in comb if f in X.columns and f not in ('Select3*-v2NA', 'Select6*-v2NA')]
    if "Select3*-v2NA" in comb:
        c += ["Select3*-v2NA_0.0", "Select3*-v2NA_1.0", "Select3*-v2NA_NA"]
    if "Select6*-v2NA" in comb:
        c += ["Select6*-v2NA_0.0", "Select6*-v2NA_1.0", "Select6*-v2NA_NA"]
    comb = c
    _X = X[comb].copy()
    # Remove rows with missing values in columns from comb
    _X = _X.dropna()
    y = y[_X.index]
    # Only keep the with index in _X.index
    X = X.loc[_X.index]
    res_mcdonald = combined_ftest_5x2cv_balanced_accuracy_with_threshold(estimator1=model,
                                                                         estimator2=BaselineMcDonald(),
                                                                         threshold1=threshold,
                                                                         threshold2=0.5,
                                                                         comb1=comb,
                                                                         comb2=["Thompson"],
                                                                         X=X, y=y,
                                                                         random_seed=RANDOM_SEED)

    return res_mcdonald


def compare_all_models_with_the_baseline(X, y, models_csv_path, pipeline=False):
    models_outperforming_mcdonald = []
    all_models_regardless_of_performance = []

    models_df = pd.read_csv(models_csv_path)
    # models_df = remove_models_using_only_one_binary_variable(models_df)

    j = 0

    for i, row in models_df.iterrows():
        print(f"Progress: {j + 1}/{len(models_df)}")
        j += 1
        model = row["model"]
        features = [f for f in row.index if f not in ("model", "threshold") and row[f]]
        model = model + '()' if not model.endswith('()') else model
        if model == 'SVC()':
            model = 'SVC(probability=True)'
        if pipeline:
            model = Pipeline([("scaler", StandardScaler()), ("model", eval(model))])
        else:
            model = eval(model)

        res_mcdonald = compare_model_with_baseline(model, row["threshold"], features, X, y)

        if 'XGB' in str(model):
            model = 'XGBClassifier()'

        infos = {
            "model": model,
            "threshold": row["threshold"],
            "avg_balanced_accuracy": res_mcdonald["mean_score_est1"],
            "mcdonald_pval": res_mcdonald["p_value"],
            "mcdonald_f_stat": res_mcdonald["f_stat"],
            "mcdonald_mean_diff": res_mcdonald["mean_diff"],
            "mcdonald_lower_ci": res_mcdonald["lower_ci"],
            "mcdonald_upper_ci": res_mcdonald["upper_ci"],
        }
        infos.update({f: f in features for f in ALL_FEATURES})

        all_models_regardless_of_performance.append(infos)

        if res_mcdonald["p_value"] < 0.05 and res_mcdonald["mean_diff"] > 0:
            infos = {
                "model": model,
                "threshold": row["threshold"],
                "avg_balanced_accuracy": res_mcdonald["mean_score_est1"],
                "mcdonald_pval": res_mcdonald["p_value"],
                "mcdonald_f_stat": res_mcdonald["f_stat"],
                "mcdonald_mean_diff": res_mcdonald["mean_diff"],
            }
            infos.update({f: f in features for f in ALL_FEATURES})
            models_outperforming_mcdonald.append(infos)

    models_outperforming_mcdonald = pd.DataFrame(models_outperforming_mcdonald)
    all_models_regardless_of_performance = pd.DataFrame(all_models_regardless_of_performance)

    if not all_models_regardless_of_performance.empty:  # This should always be the case
        # First divide into two dataframes, one with models that outperform the McDonald baseline and the other with
        # models that do not outperform the McDonald baseline
        outperforming = all_models_regardless_of_performance[all_models_regardless_of_performance["mcdonald_pval"] < 0.05]
        not_outperforming = all_models_regardless_of_performance[all_models_regardless_of_performance["mcdonald_pval"] >= 0.05]

        # Sort the dataframes by the average balanced accuracy
        outperforming = outperforming.sort_values(by=['avg_balanced_accuracy'], ascending=False)
        not_outperforming = not_outperforming.sort_values(by=['avg_balanced_accuracy'], ascending=False)

        # Concatenate the two dataframes
        all_models_regardless_of_performance = pd.concat([outperforming, not_outperforming])

        # Add a column to indicate whether the model uses only the simplified variables
        simplified_variables_only = (~all_models_regardless_of_performance['% perivenular les']) & \
                                    (~all_models_regardless_of_performance['CL-count-updated']) & \
                                    (~all_models_regardless_of_performance['number_PRL'])

        all_models_regardless_of_performance["simplified_variables_only"] = simplified_variables_only

        # Save the output dataframe
        all_models_regardless_of_performance.to_csv(ALL_MODELS_REGARDLESS_OF_PERFORMANCE_PATH, index=False)

    if models_outperforming_mcdonald.empty:
        pprint("No model outperforms the DIS McDonald baseline")
    else:
        models_outperforming_mcdonald = models_outperforming_mcdonald.sort_values(by=['avg_balanced_accuracy'],
                                                                                  ascending=False)
        simplified_variables_only = (~models_outperforming_mcdonald['% perivenular les']) & \
                                    (~models_outperforming_mcdonald['CL-count-updated']) & \
                                    (~models_outperforming_mcdonald['number_PRL'])
        models_outperforming_mcdonald["simplified_variables_only"] = simplified_variables_only

        pprint(f"{len(models_outperforming_mcdonald)} models outperform the McDonald baseline")

        models_outperforming_mcdonald.to_csv(MODELS_OUTPERFORMING_MCDONALD_PATH, index=False)

    return models_outperforming_mcdonald, all_models_regardless_of_performance


def remove_models_using_only_one_binary_variable(models_df):
    """
    Remove models that only use a single binary variable
    """

    all_variables = [f for f in ALL_FEATURES if f not in ("age", "sex")]
    binary_variables = ["OCB_presence", "Thompson", "Filippi", "Select3*-v2NA", "Select6*-v2NA", "CL1", "PRL1"]
    indices_to_remove = []
    for bin_var in binary_variables:
        indices_to_remove += models_df[(models_df[bin_var]) & (models_df[all_variables].sum(axis=1) == 1)].index.tolist()

    models_df = models_df.drop(indices_to_remove)
    return models_df


def analyze_proportions(df_models):
    df_models['CVS'] = df_models['% perivenular les'] | df_models['Select3*-v2NA'] | df_models['Select6*-v2NA']
    df_models['PRL'] = df_models['number_PRL'] | df_models['PRL1']
    df_models['CL'] = df_models['CL-count-updated'] | df_models['CL1']
    n_combinations_using_cvs = 54
    n_combinations_using_prl = 48
    n_combinations_using_cl = 48
    n_combinations_using_dis = 36
    df_models_mcdonald = df_models

    n_models_outperforming_mcdonald = len(df_models_mcdonald)
    pprint("1 - How many models outperformed McDonald's DIS criteria?")
    pprint(f"Number of models outperforming McDonald's DIS criteria: {n_models_outperforming_mcdonald}")
    pprint()

    # 2 - What is the performance of the best model outperforming McDonald's DIS criteria?
    best_model_mcdonald = df_models_mcdonald.iloc[0]
    used_feats = [f for f in ALL_FEATURES if best_model_mcdonald[f]]
    pprint("2 - What is the performance of the best model outperforming McDonald's DIS criteria?")
    pprint(f"Best model outperforming McDonald's DIS criteria used the following features: {used_feats}")
    pprint(f"Metrics for the best model outperforming McDonald's DIS criteria:")
    pprint(f"Model: {best_model_mcdonald['model']}")
    pprint(f"Balanced Accuracy: {round(best_model_mcdonald['avg_balanced_accuracy'] * 100, 1)}")
    pprint()

    # 6 - How many models using only simplified variables outperformed McDonald's DIS criteria?
    simplified_models_indices = df_models_mcdonald[df_models_mcdonald["simplified_variables_only"]].index
    n_simplified_models_outperforming_mcdonald = len(simplified_models_indices)
    pprint("6 - How many models using only simplified variables outperformed McDonald's DIS criteria?")
    pprint(
        f"Number of models using only simplified variables outperforming McDonald's DIS criteria: {n_simplified_models_outperforming_mcdonald}")
    pprint()

    # 7 - What is the best model using only simplified variables outperforming McDonald's DIS criteria?
    best_simplified_model_mcdonald = df_models_mcdonald.loc[simplified_models_indices[0]]
    used_feats = [f for f in ALL_FEATURES if best_simplified_model_mcdonald[f]]
    pprint("7 - What is the best model using only simplified variables outperforming McDonald's DIS criteria?")
    pprint(
        f"Best model using only simplified variables outperforming McDonald's DIS criteria used the following features: {used_feats}")
    pprint(f"Metrics for the best model using only simplified variables outperforming McDonald's DIS criteria:")
    pprint(f"Model: {best_simplified_model_mcdonald['model']}")
    pprint(f"Balanced Accuracy: {round(best_simplified_model_mcdonald['avg_balanced_accuracy'] * 100, 1)}")
    pprint()

    # 8 - Among the models that outperformed DIS, how many used CVS
    pprint("8 - Among the models that outperformed DIS, how many used CVS?")
    n_models_using_cvs = len(df_models_mcdonald[df_models_mcdonald["CVS"]])
    pprint(f"Number of models that outperformed DIS and used CVS: {n_models_using_cvs} / {n_combinations_using_cvs} "
           f"({round(n_models_using_cvs / n_combinations_using_cvs * 100, 1)}%)")
    pprint()

    # 9 - Among the models that outperformed DIS, how many used PRL
    pprint("9 - Among the models that outperformed DIS, how many used PRL?")
    n_models_using_prl = len(df_models_mcdonald[df_models_mcdonald["PRL"]])
    pprint(f"Number of models that outperformed DIS and used PRL: {n_models_using_prl} / {n_combinations_using_prl} "
           f"({round(n_models_using_prl / n_combinations_using_prl * 100, 1)}%)")
    pprint()

    # 10 - Among the models that outperformed DIS, how many used CL
    pprint("10 - Among the models that outperformed DIS, how many used CL?")
    n_models_using_cl = len(df_models_mcdonald[df_models_mcdonald["CL"]])
    pprint(f"Number of models that outperformed DIS and used CL: {n_models_using_cl} / {n_combinations_using_cl} "
           f"({round(n_models_using_cl / n_combinations_using_cl * 100, 1)}%)")
    pprint()

    # 11 - Among the models that outperformed DIS, how many used CL
    pprint("11 - Among the models that outperformed DIS, how many used DIS?")
    n_models_using_dis = len(df_models_mcdonald[df_models_mcdonald["Thompson"]])
    pprint(f"Number of models that outperformed DIS and used DIS: {n_models_using_dis} / {n_combinations_using_dis} "
           f"({round(n_models_using_dis / n_combinations_using_dis * 100, 1)}%)")
    pprint()

    # 12 - Among the models that outperformed DIS, which are the most frequent combinations?
    pprint("12 - Among the models that outperformed DIS, which are the most frequent combinations?")
    most_frequent_combinations = df_models_mcdonald.groupby(["PRL", "CVS", "CL", "Thompson"]).size().reset_index(
        name='counts')
    most_frequent_combinations = most_frequent_combinations.sort_values(by='counts', ascending=False)
    pprint(most_frequent_combinations)
    pprint()

    # 13 - Among the models that outperformed DIS, which are the most frequent combinations using only simplified variables?
    pprint(
        "13 - Among the models that outperformed DIS, which are the most frequent combinations using only simplified variables?")
    most_frequent_combinations_simplified = df_models_mcdonald.iloc[simplified_models_indices].groupby(
        ["PRL", "CVS", "CL", "Thompson"]).size().reset_index(name='counts')
    most_frequent_combinations_simplified = most_frequent_combinations_simplified.sort_values(by='counts',
                                                                                              ascending=False)
    pprint(most_frequent_combinations_simplified)
    pprint()


if __name__ == "__main__":
    pprint(f"Random seed set to {RANDOM_SEED}")
    pprint("*** Step 2: Find subset of algorithm-combination pairs outperforming DIS ***")
    pprint()
    X, y = get_full_dataset(TRAINING_DATA_PATH, dropna=False, return_X_y=True, prepare=True)

    # for kf in ("kf", "skf"):
    #     for pipeline in ("pipeline", "no_pipeline"):
    #         models_csv_path = os.path.join("/home/mwynen/data/DIssectMS/analysis", f"best_agorithms_per_combination_of_features_{kf}_{pipeline}.csv")
    #         compare_all_models_with_both_baselines(X, y, models_csv_path, f"_{kf}_{pipeline}", pipeline == 'pipeline')

    # Compare all models against both baselines
    beating_dis, all_models = compare_all_models_with_the_baseline(X, y, BEST_ALGORITHM_PER_COMBINATION_OF_FEATURES_PATH, pipeline=USE_PIPELINE)
    print("Done!")
    print()
    pprint(f"Number of models outperforming the McDonald baseline: {len(beating_dis)}")
    pprint()

    analyze_proportions(beating_dis)
    pprint()
    pprint()







