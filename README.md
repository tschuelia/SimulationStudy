# Simulation Study

This repo contains all scripts and data to reproduce the results related to the Gradient Boosted Trees of our simulation vs. empirical study.
If you want to reproduce the results of the CNNs, see [Johanna's repo](https://github.com/JohannaTrost/seqsharp).


## Download the input and results data
The script `download_data.py` automatically downloads the data and moves the files to the proper directories.
To run the file, run the following in your terminal: `python download_data.py`
If you want to unpack all subdirectories as well, run `python download_data.py --unpack`. This will require approx. TODO GB of disk space.

## Running the pipeline

1. Create a new conda environment using the provided `environment.yml` file:
   ```commandline
    conda env create -f environment.yml
    ```
2. Activate the environment:
    ```commandline
    conda activate simulation
    ```
3. Download and install PyPythia ([instructions](https://github.com/tschuelia/PyPythia/wiki/Installation)).
4. Download and install RAxML-NG ([instructions](https://github.com/amkozlov/raxml-ng/wiki/Installation)).
5. cd into the `scripts` directory and open the `Snakefile`. On the top of the file, just after all imports you find a
   section entitled `CONFIG`. Change all settings according to the instructions below.
6. Do a dry run to see what commands snakemake will execute:
   ```commandline
    snakemake -n --quiet
    ```
7. Finally, run the pipeline:
    ```commandline
    snakemake --cores [number of cores you want to use]
    ```

### Configuring the pipeline

To reproduce all features, use the following combinations of settings in step 5 of the above instructions:

`data_src = "../../input_data/{category}/{source_dir}"`:
For each of the following `(category, source_dir)` combinations:

- gapless DNA data: `category = dna_gapless`, `source_dir`
  in [`empirical`, `sim_jc`, `sim_hky`, `sim_gtr`, `sim_gtr_g`, `sim_gtr_g_i`]
- gapless AA data: `category = aa_gapless`, `source_dir`
  in [`empirical`, `sim_poisson`, `sim_wag`, `sim_lg`, `sim_lg_c60`, `sim_lg_s0256`, `sim_lg_s0256_g4`, `sim_lg_s0256_gc`]
- DNA data with gaps: `category=dna_with_gaps`, `source_dir`
  in [`empirical`, `sim_gtr_g_i_mimick`, `sim_gtr_g_i_sparta`]
- AA data with gaps: `category=aa_with_gaps`, `source_dir` in [`empirical`, `sim_lg_s0256_gc_sabc`]

Set `empirical` to `True` if the `source_dir` is `empirical`, otherwise set it to `False`.

Make sure to update `raxml_ng` and `predictor_path` according to your setup.

## Training the classifiers
To automatically train all classifiers run `python train_classifiers.py`. 
You can configure various run parameters, such as whether to use Optuna or not by setting the respective flags in `train_classifiers.py::__main__` (starting in Line 153).  

## Provided Data and Results

The directory `dataframes` contains the results of running the feature generation pipeline. 
This directory contains one subdirectory for each of the for data types (DNA gapless, DNA with gaps, AA gapless, AA with gaps).
Each of these subdirectories contains an Apache Parquet file with prediction features and training target (label) for each MSA of the respective data collection.

You can read a parquet file in python using pandas:
```python
import pandas as pd

df = pd.read_parquet("dataframes/dna_gapless/sim_jc.parquet")
```
The column `msa_id` contains the name of the respective MSA, and the `empirical` column is the training target.
The remaining columns contain the prediction features and the names should be self-explanatory.


The directory `training_results` contains the results of training the GBT classifier and contains the following
subdirectories:

* `2023-04-26_with_optuna_basic_brlens_randomness`: GBT classifiers trained using all features presented in the paper
* `2023-04-27_with_optuna_basic_brlens`: GBT classifiers trained without the randomness features
* `2023-04-27_with_optuna_randomness`: GBT classifiers trained without the branch length features

Each of these contains four subdirectories (one per data type) and a `results_gain.txt` file with a summary of the
training and test accuracies, as well as the gain-based feature importance.

Each subdirectory for a specific training setup und data collection contains the following directories files:
- a directory with results of each of the 10 training folds (`train_fold_i`)
  - the `lgb_model_gain.txt` contains a textual representation of the respective LGB Booster
  - the `metrics.json` contains the training and test BACC
- `{model}_all_features.pdf` is a plot of all prediction features (simulated data collection and the respective empirical data collection)
- `{model}.pdf` plots only the five most important features 
- `best_model_gain.txt` contains the best booster of the 10 training folds in textual representation
- `metrics.json` contains the average BACC across all 10 folds
- `params_tuned.json` contains the hyperparameters of the best booster of the 10 training folds

You can use the textual model representations to load a trained booster in Python like this:
```python
import lightgbm as lgb

model_file = "training_results/dna_gapless/sim_jc/best_model_gain.txt"
model = lgb.Booster(model_file=model_file)
```

### Naming of the files
The following table shows the mapping between the file names (both in the `dataframes` and `training_results` directories) and the dataset names we used in the paper.


| Dataset Name      | Paper reference   |
|-------------------|-------------------|
| DNA Gapless       |                   |
| sim_jc            | JC                |
| sim_hky           | HKY               |
| sim_gtr           | GTR               |
| sim_gtr_g         | GTR+G             |
| sim_gtr_g_i       | GTR+G+I           |
| DNA with gaps     |                   |
| mimick_sim_gtr_g_i | GTR+G+I+mimick    |
| sparta_sim_gtr_g_i | GTR+G+I+sparta    |
| AA Gapless        |                   | 
| sim_poisson_gapless | Poisson           |
| sim_wag_gapless   | WAG               |
| sim_lg_gapless    | LG                |
| sim_lg_c60_gapless | LG+C60            |
| sim_lg_s0256_gapless | LG+S256           | 
| sim_lg_s0256_g4_gapless | LG+S256+G4        |
| sim_lg_s0256_gc_gapless | LG+S256+GC        |
| AA with gaps      |                   |
| sim_lg_s0256_gc_sabc | LG+S256+GC+sparta |



## Publication
The respective publication is available as preprint on BioRxiv
TODO: add publication.