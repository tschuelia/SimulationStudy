import json
import logging
import os
import pathlib
from functools import partial
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from tabulate import tabulate

logger = logging.getLogger("training")
optuna.logging.set_verbosity(optuna.logging.WARNING)


MSA_FEATURES = [
    "difficulty",
    "num_patterns/num_taxa",
    "num_sites/num_taxa",
    "proportion_gaps",
    "proportion_invariant",
    "entropy",
    "bollback",
    "avg_rfdist_parsimony",
    "proportion_unique_topos_parsimony",
    "num_patterns/num_sites",
    "pattern_entropy"
]

RANDOMNESS_FEATURES = [
    "entropy_rand",
    "chi_2",
    "mean_rand",
    "mcpi",
    "scc"
]

BASIC_BRLEN_STATS = [
    "mean__all_brlens",
    "median__all_brlens",
    "stdev__all_brlens",
    "total__all_brlens",
    "min__all_brlens",
    "max__all_brlens",
]

FEATURES_BASIC_BRLENS = list(set(MSA_FEATURES + BASIC_BRLEN_STATS))

EXTENDED_BRLEN_STATS = [
    "mean__external_brlens",
    "median__external_brlens",
    "stdev__external_brlens",
    "total__external_brlens",
    "min__external_brlens",
    "max__external_brlens",
    "mean__internal_brlens",
    "median__internal_brlens",
    "stdev__internal_brlens",
    "total__internal_brlens",
    "min__internal_brlens",
    "max__internal_brlens",
    "mean__all_brlens",
    "median__all_brlens",
    "stdev__all_brlens",
    "total__all_brlens",
    "min__all_brlens",
    "max__all_brlens",
]

FEATURES_EXTENDED_BRLENS = list(set(MSA_FEATURES + EXTENDED_BRLEN_STATS))

LABEL = "empirical"

# =========================
# Gapless data
# =========================
SIMULATED_DNA_DATA_GAPLESS = [
    "sim_jc",
    "sim_hky",
    "sim_gtr",
    "sim_gtr_g",
    "sim_gtr_g_i",
]

SIMULATED_AA_DATA_GAPLESS = [
    "alisim_poisson_gapless",
    "alisim_wag_gapless",
    "alisim_lg_gapless",
    "alisim_lg_c60_gapless",
    "alisim_lg_s0256_gapless",
    "alisim_lg_s0256_g4_gapless",
    "alisim_lg_s0256_gc_gapless",
]

# =========================
# data with gaps
# =========================
SIMULATED_DNA_DATA_WITH_GAPS = ["mimick_sim_gtr_g_i", "sparta_sim_gtr_g_i"]

SIMULATED_AA_DATA_WITH_GAPS = ["alisim_lg_s0256_gc_sabc"]

# =========================

COLOR_SIMULATED = "MediumPurple"
COLOR_EMPIRICAL = "LightSeaGreen"


def load_data(data_src: pathlib.Path, name: str) -> pd.DataFrame:
    parquet_path = data_src / (name + ".parquet")

    if not parquet_path.exists():
        raise FileNotFoundError("Parquet does not exist: ", parquet_path)

    df = pd.read_parquet(parquet_path)
    return df


def concat_data_and_correct_dtypes(
    simulated: pd.DataFrame, empirical: pd.DataFrame
) -> pd.DataFrame:
    df = pd.concat([empirical, simulated])
    for feat in FEATURES_EXTENDED_BRLENS:
        if df[feat].dtype == object:
            df[feat] = df[feat].astype(float)

    return df


def get_feature_importances(cls: lgb.Booster, features: List):
    return sorted(
        list(zip(features, cls.feature_importance(importance_type="gain").round(2))),
        key=lambda x: x[1],
        reverse=True,
    )


def get_accuracy(
    lgb_model: lgb.Booster,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[float, float, float, float]:
    y_pred_train = lgb_model.predict(X_train).round().astype(int)
    y_pred_test = lgb_model.predict(X_test).round().astype(int)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    bacc_train = balanced_accuracy_score(y_train, y_pred_train)
    bacc_test = balanced_accuracy_score(y_test, y_pred_test)

    return acc_train, acc_test, bacc_train, bacc_test


def save_model(
    model: lgb.Booster, outdir: pathlib.Path, model_prefix: str = "lgb_model"
) -> None:
    for importance_type in ["split", "gain"]:
        model_file = outdir / f"{model_prefix}_{importance_type}.txt"
        model.save_model(model_file, importance_type=importance_type)


def save_results(
    acc_train: float,
    acc_test: float,
    bacc_train: float,
    bacc_test: float,
    outdir: pathlib.Path,
) -> None:
    data = {
        "accuracy training": acc_train,
        "accuracy test": acc_test,
        "balanced accuracy training": bacc_train,
        "balanced accuracy test": bacc_test,
    }

    with (outdir / "metrics.json").open("w") as f:
        json.dump(data, f)


def stratified_training(
    X: pd.DataFrame,
    y: pd.Series,
    lgb_params: Dict,
    store_output: bool = False,
    outdir_base: pathlib.Path = None,
) -> List:
    if store_output and outdir_base is None:
        raise ValueError("Provide an outdir if you want to store the model outputs")

    strat_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    splits = strat_kfold.split(X, y)

    results = []

    for i, (train_index, test_index) in enumerate(splits):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        lgb_dataset = lgb.Dataset(X_train, label=y_train)
        lgb_model = lgb.train(params=lgb_params, train_set=lgb_dataset)

        acc_train, acc_test, bacc_train, bacc_test = get_accuracy(
            lgb_model, X_train, X_test, y_train, y_test
        )

        results.append((lgb_model, acc_train, acc_test, bacc_train, bacc_test))

        if store_output:
            outdir = outdir_base / f"train_fold_{i}"
            os.makedirs(outdir, exist_ok=True)
            save_model(lgb_model, outdir)
            save_results(acc_train, acc_test, bacc_train, bacc_test, outdir)

    return results


def get_raw_figures_for_col(
    col: str, simulated: pd.DataFrame, empirical: pd.DataFrame, showlegend: bool = True
) -> Tuple[go.Histogram, go.Histogram]:
    combined = pd.concat([simulated[col], empirical[col]])

    lower = combined.quantile(0.1)
    upper = combined.quantile(0.9)

    simu = simulated.loc[simulated[col].between(lower, upper)][col]
    empi = empirical.loc[empirical[col].between(lower, upper)][col]

    fig_simu = go.Histogram(
        x=simu,
        histnorm="percent",
        name="simulated",
        marker_color=COLOR_SIMULATED,
        showlegend=showlegend,
        nbinsx=60,
    )

    fig_empi = go.Histogram(
        x=empi,
        histnorm="percent",
        name="empirical",
        marker_color=COLOR_EMPIRICAL,
        showlegend=showlegend,
        nbinsx=60,
    )

    return fig_simu, fig_empi


def plot_cols(
    columns: List[str],
    simulated: pd.DataFrame,
    empirical: pd.DataFrame,
    title: str = "",
) -> go.Figure:
    fig = make_subplots(rows=len(columns), cols=1, subplot_titles=columns)

    for i, col in enumerate(columns):
        fig_simu, fig_empi = get_raw_figures_for_col(
            col, simulated, empirical, showlegend=i == 0
        )
        fig.append_trace(fig_simu, row=i + 1, col=1)
        fig.append_trace(fig_empi, row=i + 1, col=1)

        fig.update_xaxes(title=col, row=i + 1, col=1)
    fig.update_yaxes(title="proportion", ticksuffix="%")
    fig.update_layout(
        template="plotly_white", height=300 * len(columns), width=2000, title=title
    )
    return fig


def lgb_binary_objective(trial, lgb_dataset, n_folds):
    params_tune = {
        "objective": "binary",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25),
        "max_depth": trial.suggest_int("max_depth", 5, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 20),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 30, 100),
        "verbose": -1,
        "feature_pre_filter": False,
    }

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    obj_value = lgb.cv(params_tune, lgb_dataset, folds=kfold,)[
        "binary_logloss-mean"
    ][-1]
    return obj_value


def optuna_optimization(
    X: pd.DataFrame, y: pd.Series, n_trials: int = 100
) -> optuna.Study:
    params = {
        "learning_rate": 0.05,
        "max_depth": 10,
        "lambda_l1": 1e-6,
        "lambda_l2": 0.1,
        "num_leaves": 20,
        "bagging_fraction": 0.9,
        "bagging_freq": 4,
        "min_child_samples": 50,
        "verbose": -1,
        "num_iterations": 1000,
    }

    n_classes = y.value_counts().shape[0]

    print("PARAMS", params)

    lgb_dataset = lgb.Dataset(X, label=y)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler()
    )
    study.enqueue_trial(params)
    partial_objective = partial(
        lgb_binary_objective, lgb_dataset=lgb_dataset, n_folds=10
    )
    study.optimize(partial_objective, n_trials=n_trials)

    return study


def run_optuna_study(
    X: pd.DataFrame, y: pd.Series, n_trials: int = 100
) -> Dict:
    study = optuna_optimization(X, y, n_trials=n_trials)
    params_tuned = study.best_trial.params

    return params_tuned


def read_metrics(
    metrics_file: pathlib.Path, num_decimals: int = 3
) -> Tuple[float, float, float, float]:
    data = json.load(metrics_file.open())

    acc_train = round(data["accuracy training"], num_decimals)
    acc_test = round(data["accuracy test"], num_decimals)
    bacc_train = round(data["balanced accuracy training"], num_decimals)
    bacc_test = round(data["balanced accuracy test"], num_decimals)

    return acc_train, acc_test, bacc_train, bacc_test


def get_n_most_important_features_for_model_file(
    model_file: pathlib.Path, features: List[str], n_features: int = 5
) -> List:
    best_model = lgb.Booster(model_file=model_file)

    feature_importances = get_feature_importances(best_model, features)
    return [feat for feat, _ in feature_importances[:n_features]]


def get_results_table(
    result_base_dir: pathlib.Path,
    simulated_data_names: List[str],
    features: List[str],
    feat_importance_type: str,
) -> str:
    headers = [
        "DATA",
        "avg ACC (train)",
        "avg ACC (test)",
        "avg BACC (train)",
        "avg BACC (test)",
        f"Top 5 important features (using best predictor out of 10; feature importance = {feat_importance_type})",
    ]
    table = []

    for simu in simulated_data_names:
        outdir_base = result_base_dir / simu

        metrics_file = outdir_base / "metrics.json"
        acc_train, acc_test, bacc_train, bacc_test = read_metrics(
            metrics_file, num_decimals=3
        )

        model_file = outdir_base / f"best_model_{feat_importance_type}.txt"

        top5_importances = get_n_most_important_features_for_model_file(
            model_file, features, n_features=5
        )
        top5_importances = ", ".join([el.strip("'") for el in top5_importances])
        table.append(
            [simu, acc_train, acc_test, bacc_train, bacc_test, top5_importances]
        )

    return tabulate(table, headers, tablefmt="github")


def plot_important_features(
    data_src_dir: pathlib.Path,
    result_base_dir: pathlib.Path,
    simulated_data_names: List[str],
    empirical_data_name: str,
    features: List[str],
    feat_importance_type: str,
    n_features: int = 5,
) -> None:
    for simu in simulated_data_names:
        simulated_df = load_data(data_src_dir, simu)
        empirical_df = load_data(data_src_dir, empirical_data_name)

        outdir_base = result_base_dir / simu

        model_file = outdir_base / f"best_model_{feat_importance_type}.txt"
        important_features = get_n_most_important_features_for_model_file(
            model_file, features, n_features=n_features
        )

        plot_file = outdir_base / (simu + ".pdf")
        logger.info(f"Plotting features for {simu} -> {plot_file}")

        fig = plot_feature_distributions(
            simulated_df=simulated_df,
            empirical_df=empirical_df,
            features=important_features,
            title=f"{simu} ({n_features} most important features (type: {feat_importance_type}))",
        )
        fig.write_image(plot_file)


def plot_all_features(
    data_src_dir: pathlib.Path,
    result_base_dir: pathlib.Path,
    simulated_data_names: List[str],
    empirical_data_name: str,
) -> None:
    for simu in simulated_data_names:
        simulated_df = load_data(data_src_dir, simu)
        empirical_df = load_data(data_src_dir, empirical_data_name)

        outdir_base = result_base_dir / simu

        plot_file = outdir_base / (simu + "_all_features.pdf")
        logger.info(f"Plotting features for {simu} -> {plot_file}")

        fig = plot_feature_distributions(
            simulated_df=simulated_df,
            empirical_df=empirical_df,
            features=FEATURES_EXTENDED_BRLENS,
            title=f"{simu} (all features)",
        )
        fig.write_image(plot_file)


def get_lower_and_upper_for_plots(data: pd.Series) -> Tuple[float, float]:
    if min(data) == 0:
        lower = 0
    else:
        lower = data.quantile(0.1)
    if min(data) == 1:
        upper = 1
    else:
        upper = data.quantile(0.9)

    return lower, upper


def plot_feature_distributions(
    simulated_df: pd.DataFrame,
    empirical_df: pd.DataFrame,
    features: List[str],
    title: str,
):
    fig = make_subplots(rows=len(features), cols=1, subplot_titles=features)

    for i, feat in enumerate(features):
        all_data = simulated_df[feat] + empirical_df[feat]
        lower, upper = get_lower_and_upper_for_plots(all_data)

        simu_data = simulated_df.loc[simulated_df[feat].between(lower, upper)][feat]
        empi_data = empirical_df.loc[empirical_df[feat].between(lower, upper)][feat]

        fig.append_trace(
            go.Histogram(
                x=simu_data,
                histnorm="percent",
                name="simulated",
                marker_color=COLOR_SIMULATED,
                showlegend=i == 0,
            ),
            row=i + 1,
            col=1,
        )

        fig.append_trace(
            go.Histogram(
                x=empi_data,
                histnorm="percent",
                name="empirical",
                marker_color=COLOR_EMPIRICAL,
                showlegend=i == 0,
            ),
            row=i + 1,
            col=1,
        )

        fig.update_xaxes(title=feat, row=i + 1, col=1)

    fig.update_yaxes(title="Proportion", ticksuffix="%")
    fig.update_layout(
        title=title, height=300 * len(features), width=1000, template="plotly_white"
    )
    return fig
