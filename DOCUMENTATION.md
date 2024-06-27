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
- `advanced_hurdle_models.ipynb`:
- `autoML.ipynb`:
- `basic_hurdle_model.ipynb`:
- `cross_validation.ipynb`:
- `data_exploration.ipynb:`
- `designing_two_stage_novel_model.ipynb`:
- `dl_embeddings.ipynb`:
- `extracting_species_traits.ipynb`:
- `extreme_generalisation.ipynb`:
- `loading_spatial_data.ipynb`:
- `visualizing_results.ipynb`:

**Miscellaneous:**
- `gee_scripts/`:
