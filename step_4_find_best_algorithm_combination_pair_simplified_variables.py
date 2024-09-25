from utils import *


if __name__ == "__main__":
    pprint("*** Step 4: Find the best algorithm combination pair using simplified variables ***")
    pprint()
    X, y = get_full_dataset(TRAINING_DATA_PATH, dropna=False, return_X_y=True, prepare=True)
    models_outperforming_baseline = pd.read_csv(MODELS_OUTPERFORMING_MCDONALD_PATH)
    models_outperforming_baseline = models_outperforming_baseline[models_outperforming_baseline["simplified_variables_only"]]

    indices = models_outperforming_baseline.index

    results = compare_model_with_other_models(X, y, models_outperforming_baseline, index=0, pipeline=USE_PIPELINE)
    used_feats = [f for f in ALL_FEATURES if models_outperforming_baseline.loc[indices[0]][f]]
    pprint()
    pprint(f"There are {len(indices)} models outperforming the baseline model, while using only simplified variables.")
    pprint(f"The best model is model {indices[0]}. {used_feats}")
    for i, res in enumerate(results):
        diff = res["pval"] > 0.05
        if not diff: break

        sign = "not significantly different from" if diff else "significantly different from"
        pprint(f"Model {indices[i+1]} is {sign} the best model. p-value: {res['pval']}, mean diff: {res['mean_diff']}", end= " ")
        used_feats = [f for f in ALL_FEATURES if models_outperforming_baseline.loc[indices[i+1]][f]]
        pprint(used_feats)
    pprint()
    pprint()