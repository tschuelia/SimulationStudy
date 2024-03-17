# Simulation Study

This repo contains all scripts and data to reproduce the results related to the Gradient Boosted Trees of our simulation vs. empirical study.

If you want to reproduce/use the results of the CNNs, see [Johanna's repo](https://github.com/JohannaTrost/seqsharp).
If you want to reproduce the GBT results see the file `reproducing.md`.

This README includes instructions on how to use the pipeline and `batch_prediction.py` to analyze your own MSA data collection.
Please note that our classifiers are trained on a specific set of MSAs and evolutionary models. While you can use your own data collection and one of our 
pretrained classifiers, the results are not necessarily meaningful and should be handled with care.

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
5. Download and install the ENT — Fourmilab Random Sequence Tester ([instructions](https://github.com/Fourmilab/ent_random_sequence_tester).
6. Configure the pipeline according to the instructions below.
7. Do a dry run to see what commands snakemake will execute:
   ```commandline
    snakemake -n --quiet
    ```
8. Finally, run the pipeline:
    ```commandline
    snakemake --cores [number of cores you want to use]
    ```

## Pipeline configuration
To set up the pipeline for your own data collection, cd into the `scripts` directory, 
open the `config.yaml` and set the variables according to the provided description.


## Predicting empirical vs. simulated for your data
The final result of the above pipeline will be a pandas dataframe in `{outdir}/{parquet_file_prefix}.parquet` according to the configured settings above.
Open the `batch_prediction.py` file and change the `CLASSIFIER_PATH` to a filepath of the pretrained LGB classifier you want to use. 
Set `FEATURES_DATAFRAME` to the pipeline results `{outdir}/{parquet_file_prefix}.parquet`.
Finally, run the prediction: `python batch_prediction.py`.
This will classify all MSAs based on the provided features, and output a summary of the predictions and the updated dataframe 
with all predictions in `predictions/{parquet_file_prefix}.parquet`.


## Publication
The paper explaining the details of our simulation study is published in MBE:    
Trost, J.<sup>1</sup>, Haag, J.<sup>1</sup>, Höhler, D.<sup>1</sup>, Nesterenko, L., Jacob, L., Stamatakis, A. & Boussau, B. (2024). **Simulations of Sequence Evolution: How (Un)realistic They Are and Why.** *Molecular Biology and Evolution*, 41(1). [https://doi.org/10.1093/molbev/msad277](https://doi.org/10.1093/molbev/msad277)
<br><sup>1</sup>equal contribution


