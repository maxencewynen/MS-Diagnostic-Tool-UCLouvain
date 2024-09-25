from utils import *

if __name__ == "__main__":
    pprint("*** Step 5: Compare the best overall model with the best simplified model ***")
    pprint()
    X, y = get_full_dataset(TRAINING_DATA_PATH, dropna=False, return_X_y=True, prepare=True)
    models_outperforming_baseline = pd.read_csv(MODELS_OUTPERFORMING_MCDONALD_PATH)
    simplified_models = models_outperforming_baseline[models_outperforming_baseline["simplified_variables_only"]]

    best_simplified_model_index = simplified_models.index[0]

    results = compare_model_with_other_models(X, y, models_outperforming_baseline, index=0,
                                              index2=best_simplified_model_index,pipeline=USE_PIPELINE)
    res = results[0]
    diff = res["pval"] > 0.05

    sign = "not significantly different from" if diff else "significantly different from"
    pprint(f"The best simplified model is {sign} the best model. p-value: {res['pval']}, mean diff: {res['mean_diff']}")
    pprint()