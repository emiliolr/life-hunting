# Repository Documentation

## Local setup

For local setup, I recommend using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) or [`mamba`](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html); if you choose to use `mamba`, simply replace `conda` in all commands with `mamba`. The steps to get set up locally are:
1. Clone this GitHub repository. Be sure to use `git clone --recursive <HTTPS or SSH>` to ensure the `satclip` submodule is properly loaded.
2. Navigate to the base directory of your local version of the repository.
3. Open `environment.yml` and replace the "prefix" field with the directory to your `conda` or `mamba` installation.
  - This can be found using `conda info --envs`: copy the path to the base environment and append `envs/life-hunting`.
4. Run `conda env create -f environment.yml`.
5. Activate the newly created environment using `conda activate life-hunting`.
6. Run `pip install -r requirements.txt` to install the final dependencies.
7. Replace the filepaths in `config.json`; the only two that are of importance for reproducing the results from the MRes report are the "benitez_lopez2019" and "ferreiro_arias2024" filepaths.
  - See the repository [README](README.md) for details on how to download these datasets.

## Repository structure

For the most part, functionality is implemented in Python files (`*.py`) and experiments are run through Python notebooks (`*.ipynb`). Some notebooks were used to explore initial ideas that ultimately did not make it into the report. A brief description of each file is included below; please refer to each individual file for additional details.

**Python files:**
These are all in the base repository directory.
- `cross_validation.py`: The main functionality for cross-validation experiment, including spatial- and species-blocking. For an example of how to use these functions, see `cross_validation.ipynb`.
- `custom_metrics.py`: Several metrics that are not implemented directly in `sklearn`.
- `embeddings.py`: Functionality for obtaining species and spatial embeddings using [BioCLIP](https://imageomics.github.io/bioclip/) and [SatCLIP](https://github.com/microsoft/satclip), respectively.
- `model_utils.py`: Wrapper classes to facilitate two-stage hurdle modeling and a basic cross-validation function for use in reproducing the previous state-of-the-art.
- `performance_reports.py`: Several convenience functions to save and/or display key metrics and plots of interest, which was used primarily for rapid model prototyping.
- `plotting_utils.py`: A few common plots, again used for rapid model prototyping.
- `utils.py`: Miscellaneous utility functions. Of particular importance are `preprocess_data` and `get_zero_nonzero_datasets`.

**Python notebooks:**
These are all located in the `notebooks/` directory:
- `advanced_hurdle_models.ipynb`: Initial experimentation for a fixed-effects hurdle model using regularized logistic regression and regularized linear regression for the zero and nonzero components, respectively.
- `autoML.ipynb`: Initial experimentation for automated machine learning using `FLAML`. Direct regression and classification are considered, as well as the hurdle model structure.
- `basic_hurdle_model.ipynb`: Experiments for reproducing the previous state-of-the-art, a two-stage mixed-effects generalized linear hurdle model, implemented using `Pymer4` which directly accesses the `lme4` R package.
- `cross_validation.ipynb`: All cross-validation experiments for all models, both datasets, and all blocking methods.
- `data_exploration.ipynb`: Exploratory data analysis for the two datasets considered.
- `designing_two_stage_novel_model.ipynb`: Initial experimentation for a novel two-stage model. This ultimately wasn't included in the MRes report due to time constraints.
- `dl_embeddings.ipynb`: Initial exploration of deep learning species and spatial embeddings.
- `extracting_species_traits.ipynb`: Initial exploration of methods to extract additional species predictors through auxiliary datasets. This ultimately wasn't included in the MRes report due to time constraints.
- `extreme_generalisation.ipynb`: Experiments for extreme taxonomic and spatial model generalization, with a focus on the linear and nonlinear hurdle models.
- `loading_spatial_data.ipynb`: Initial exploration of methods for interfacing with Google Earth Engine to extract additional spatial predictors. This ultimately wasn't included in the MRes report due to time constraints.
- `visualizing_results.ipynb`: Plotting code to visualize results from both cross-validation and extreme generalization experiments.

**Miscellaneous:**
- `gee_scripts/`: Copies of the Google Earth Engine scripts used to extract additional spatial predictors. This ultimately wasn't included in the MRes report due to time constraints. Please see the [`README`](gee_scripts/README.md) in the folder for additional details.
