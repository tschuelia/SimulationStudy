import datetime
import textwrap
import warnings

import tqdm.contrib.concurrent

from simu_prediction_utils import *

warnings.filterwarnings("ignore")


def get_best_params(
    outdir: pathlib.Path, X: pd.DataFrame, y: pd.Series, rerun: bool
) -> dict:
    params_out = outdir / "params_tuned.json"

    if params_out.exists() and not rerun:
        logger.info("Best params already computed, loading")
        return json.load(open(params_out))
    else:
        logger.info("Running optuna to determine best params")
        params_tuned = run_optuna_study(X, y)
        with params_out.open("w") as f:
            json.dump(params_tuned, f)

        return params_tuned


def optimize_and_train(args) -> None:
    data_src: pathlib.Path
    outdir_base: pathlib.Path
    simulated_data_name: str
    use_optuna: bool
    rerun: bool
    features: List[str]
    label: str
    (
        data_src,
        outdir_base,
        simulated_data_name,
        use_optuna,
        rerun,
        features,
        label,
    ) = args

    empirical_df = load_data(data_src, "empirical")
    try:
        simulated_df = load_data(data_src, simulated_data_name)
    except FileNotFoundError:
        logger.warning(
            f"File for dataset {simulated_data_name} does not exist, skipping"
        )
        return

    outdir = outdir_base / simulated_data_name

    if outdir.exists() and not rerun:
        logger.warning(
            f"refusing to overwrite existing trainings, skipping {simulated_data_name}"
        )
        return
    else:
        outdir.mkdir(exist_ok=True, parents=True)

    print(f"RUNNING {simulated_data_name}")

    # 0. concat the simulated and the empirical data
    df = concat_data_and_correct_dtypes(simulated_df, empirical_df)

    X = df[features]
    y = df[label]

    # 1. Determine the best params for this binary classification
    if use_optuna:
        params = get_best_params(outdir, X, y, rerun)
    else:
        params = {
            "objective": "binary",
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

    # 2. train the classifier
    training_results = stratified_training(
        X=X, y=y, lgb_params=params, store_output=True, outdir_base=outdir
    )

    # 3. determine the best model and save it in an extra file
    # best result according to training balanced accuracy
    best_results = sorted(training_results, key=lambda x: x[2])[-1]
    best_model = best_results[0]
    save_model(best_model, outdir, model_prefix="best_model")
    save_results(*best_results[1:], outdir=outdir)


def print_res(data: list) -> None:
    results_string = ""

    for simulated, src in data:
        results_string += textwrap.dedent(
            f"""
            ==================================
            {src.name.replace("_", " ").upper()}
            ==================================
            """
        )
        results_string += get_results_table(
            result_base_dir=RESULTS,
            simulated_data_names=simulated,
            features=FEATURES,
            feat_importance_type=feat_importance_type,
        )

    print(results_string)

    with (RESULTS / f"results_{feat_importance_type}.txt").open("w") as f:
        f.write(results_string)


def plot_top_features(data: list) -> None:
    for simulated, src in data:
        plot_important_features(
            data_src_dir=src,
            result_base_dir=RESULTS,
            simulated_data_names=simulated,
            empirical_data_name="empirical",
            features=FEATURES,
            feat_importance_type=feat_importance_type,
            n_features=n_plot_features,
        )


def plot_brlen_features(data: list) -> None:
    for simulated, src in data:
        plot_all_features(
            data_src_dir=src,
            result_base_dir=RESULTS,
            simulated_data_names=simulated,
            empirical_data_name="empirical",
        )


if __name__ == "__main__":
    # if True, overwrites previous trainings for the same day (files will have a timestamp associated)
    rerun = False

    # if True, runs 100 iterations of Optuna to determine the optimal setting of hyperparameters
    use_optuna = True

    # if True, use additional basic branch length statistics (summarizing all branches in the tree)
    # if False, only use the MSA features
    use_brlens = True

    # if True, additionally separates the Branch Lengths statistics into internal, external branch lenghts
    # external branches are branches adjacent to a leaf node
    # internal branches are all other branches in the tree
    extended_brlens = False

    # if True, additionally use features based on the randomness tests
    use_randomness_features = True

    # if True, plots the n_plot_features most important features, as well as all features
    plot_features = True
    n_plot_features = 5

    # possible feature importance types are "gain" and "split"
    feat_importance_type = "gain"

    FEATURES = MSA_FEATURES

    dirname = str(datetime.date.today())
    if use_optuna:
        dirname += "_with_optuna"
    else:
        dirname += "_no_optuna"

    if extended_brlens:
        use_brlens = False  # make sure we are not appending to much information
        FEATURES = FEATURES_EXTENDED_BRLENS
        dirname += "_extended_brlens"

    if use_brlens:
        FEATURES = FEATURES_BASIC_BRLENS
        dirname += "_basic_brlens"

    if use_randomness_features:
        FEATURES.extend(RANDOMNESS_FEATURES)
        dirname += "_randomness"

    # =========================
    # Run Training
    # =========================
    print("FEATURES: ", FEATURES)

    DATA_SRC = pathlib.Path("../dataframes")
    RESULTS = pathlib.Path("../training_results") / str(dirname)
    RESULTS.mkdir(exist_ok=True, parents=True)
    print("WRITING DATA TO: ", RESULTS)

    data = [
        (SIMULATED_DNA_DATA_GAPLESS, DATA_SRC / "dna_gapless"),
        (SIMULATED_DNA_DATA_WITH_GAPS, DATA_SRC / "dna_with_gaps"),
        (SIMULATED_AA_DATA_GAPLESS, DATA_SRC / "aa_gapless"),
        (SIMULATED_AA_DATA_WITH_GAPS, DATA_SRC / "aa_with_gaps"),
    ]

    for (simulated, src) in data:
        if len(simulated) > 0:
            args = [
                (
                    src,
                    RESULTS,
                    simulated_name,
                    use_optuna,
                    rerun,
                    FEATURES,
                    LABEL,
                )
                for simulated_name in simulated
            ]

            tqdm.contrib.concurrent.process_map(optimize_and_train, args, max_workers=3)

    # =========================
    # Print results
    # =========================
    print_res(data)

    # =========================
    # Plot results
    # =========================
    if plot_features:
        plot_top_features(data)
        plot_brlen_features(data)
