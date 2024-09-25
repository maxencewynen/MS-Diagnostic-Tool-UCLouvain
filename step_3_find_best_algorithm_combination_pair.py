from utils import *


if __name__ == "__main__":
    pprint("*** Step 3: Find the best algorithm combination pair ***")
    X, y = get_full_dataset(TRAINING_DATA_PATH, dropna=False, return_X_y=True, prepare=True)
    models_outperforming_baseline = pd.read_csv(MODELS_OUTPERFORMING_MCDONALD_PATH)

    results = compare_model_with_other_models(X, y, models_outperforming_baseline, index=0, pipeline=USE_PIPELINE)
    for i, res in enumerate(results):
        diff = res["pval"] > 0.05
        if not diff: break

        sign = "not significantly different from" if diff else "significantly different from"
        pprint(f"Model {i+1} is {sign} the best model. p-value: {res['pval']}, mean diff: {res['mean_diff']}")
    pprint()