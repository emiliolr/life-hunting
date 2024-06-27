# Quantifying Species-Specific Abundance Responses to Hunting Pressure

[![DOI](https://zenodo.org/badge/777825592.svg)](https://zenodo.org/doi/10.5281/zenodo.12571509) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Project Description

In this project, I present a comprehensive assessment of approaches for predicting how local species abundance will respond to hunting pressure. In particular, I reproduced the previous state-of-the art (a mixed-effects generalised linear hurdle model), thoroughly tested (nonlinear) predictive methods through application of automated machine learning, experimented with embeddings from pre-trained deep learning models as a supplement to existing spatial and species predictors, and closely inspected spatial and taxonomic generalisability using cross-validation. I found that nonlinear hurdle models tend to outperform the existing mixed-effects linear hurdle model baseline, especially when random effects are excluded during prediction. Deep learning embeddings were largely unhelpful as supplemental predictors, but could be used to reliably predict hunting pressure when used on their own in conjunction with the nonlinear hurdle model. Finally, spatial and taxonomic generalisation remained very difficult for all models tested, but improved in the presence of more training data. Through this work, I advance the state-of-the-art for this task and provide well-documented, reproducible code to support further predictive benchmarking for this task.

This work was carried out as my Master of Research (MRes) project for the [Artificial Intelligence for Environmental Risks](https://ai4er-cdt.esc.cam.ac.uk/) Centre for Doctoral Training (AI4ER CDT). Please see my full report **(ADD REPORT PDF)** for further details on the methodology and results.

## Documentation

For a high-level overview of the structure of the repository, please see [`DOCUMENTATION.md`](DOCUMENTATION.md); this file covers local setup for the repository, contains a description of the uses of each script or notebook, and details the relevant notebook to use to reproduce figures from the report.

Each file is thoroughly documented and should be relatively self-explanatory. Python notebooks (`*.ipynb`) include markdown cells with headers describing each section's functionality and are relatively well commented. Python files (`*.py`) contain substantial documentation in the form of function docstrings; each function includes a short description of the implemented functionality and explanation of all function parameters/returns.

-----

## Acknowledgements

I would like to thank my supervisors, Tom Swinfield and Andrew Balmford, for their guidance and insights throughout the project. I would also like to thank the AI4ER support staff, Annabelle Scott and Adriana Dote, for keeping the CDT running smoothly and for their support throughout the MRes year.

-----

## License and Citation

If you use the code in this repository, please consider citing it; see the [`CITATION.cff`](CITATION.cff) file or use the "Cite this repository" function on the right sidebar. All code is under the MIT license; see the [`LICENSE`](LICENSE) for further details.

-----

## Data Availability

### Hunting datasets

**Tropical birds:** [full dataset](https://github.com/IagoFerreiroArias/Bird_Defaunation/blob/main/Data/Bird_RR_data.csv) and [corresponding publication](https://doi.org/10.1111/ddi.13855).

**Tropical mammals:** [full dataset](https://doi.org/10.6084/m9.figshare.6815288.v1) and [corresponding publication](https://doi.org/10.1371/journal.pbio.3000247).

### Deep learning embeddings

**BioCLIP:** [model weights](https://doi.org/10.57967/hf/1511) and [project site](https://imageomics.github.io/bioclip/).

**SatCLIP:** [model weights](https://huggingface.co/microsoft/SatCLIP-ResNet50-L40) and [project repository](https://github.com/microsoft/satclip).

-----

<p align="middle">
  <a href="https://ai4er-cdt.esc.cam.ac.uk/"><img src="assets/ai4er_logo.png" width="15%"/></a>
  <a href="https://www.cam.ac.uk/"><img src="assets/cambridge_logo.png" width="56%"/></a>
</p>
