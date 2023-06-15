import lightgbm as lgb
import pandas as pd
import pathlib

# ================================
# CONFIG

# Filepath pointing to a trained lightgbm predictor in txt format
# e.g. to use the trained DNA gapless GTR+G+I classifier pass
# "../training_results/2023-04-26_with_optuna_basic_brlens_randomness/dna_gapless/sim_gtr_g_i/best_model_gain.txt"
CLASSIFIER_PATH = "../training_results/2023-04-26_with_optuna_basic_brlens_randomness/dna_gapless/sim_gtr_g_i/best_model_gain.txt"

# Filepath pointing to the dataframe containing the features for all MSAs you want to classify
# this script assumes the data to be a parquet data file (same as the resulting data from our Snakemake pipeline)
# e.g. to load the GTR+G+I data frame:  "../dataframes/dna_gapless/sim_gtr_g_i.parquet"
# FEATURES_DATAFRAME = "../dataframes/dna_gapless/sim_gtr_g_i.parquet"
FEATURES_DATAFRAME = "../dataframes/027/027.parquet"
# ================================
print("CLASSIFIER: ", CLASSIFIER_PATH)
CLASSIFIER = lgb.Booster(model_file=CLASSIFIER_PATH)
FEATURES = CLASSIFIER.feature_name()

FEATURES_DATAFRAME = pathlib.Path(FEATURES_DATAFRAME)
print("MSA FEATURES: ", FEATURES_DATAFRAME)

# due to an old naming schema of the features, we might have to rename some features
renaming = {
    "pattern_entropy_mock": "pattern_entropy",
    "ent": "entropy_rand",
    "chi": "chi_2",
    "mean": "mean_rand",
    "mc_pi": "mcpi",
    "corr": "scc"
}

FEATURES = [renaming.get(f, f) for f in FEATURES]

msa_features = pd.read_parquet(FEATURES_DATAFRAME)

if not all(f in msa_features.columns for f in FEATURES):
    raise ValueError(f"Not all required features present in dataframe {FEATURES_DATAFRAME}. "
                     f"Missing features: {', '.join([f for f in FEATURES if f not in msa_features.columns])}")

print(f"Classifying {msa_features.shape[0]} MSAs.")

predictions = CLASSIFIER.predict(msa_features[FEATURES]).round().astype(bool)
msa_features["prediction"] = predictions

# count the number and proportion of wrong predictions
wrong_prediction = msa_features.loc[lambda x: x.empirical != x.prediction]
print(f"* Number of wrong predictions: {wrong_prediction.shape[0]} / {msa_features.shape[0]}")
print(f"* Fraction of wrong predictions: {100 * round(wrong_prediction.shape[0] / msa_features.shape[0], 2)}%")

# store all predictions in a parquet file
results_dir = pathlib.Path("predictions")
results_dir.mkdir(exist_ok=True)
results_file = results_dir / FEATURES_DATAFRAME.name
msa_features.to_parquet(results_file)
print("Predictions and features written to ", results_file)