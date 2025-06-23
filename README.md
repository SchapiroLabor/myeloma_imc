# myeloma_imc
Repo for reproducing the spatial analysis on the myeloma cohort from NTNU (Standal Group) for the manuscript:

## Raw data
Raw data will be uploaded to:

Raw data includes:

- Images, segmentation masks, bone masks, steinbock raw outputs, patient metadata
- Intermediary files for phenotyping (e.g. scimap gates, scimap outputs, manually annotated and aggregated batch-corrected intensities CSVs, XGB models including performances and parameters, holdout test data for xgboost) and other models like cellcharter AutoK and trVAE models
- Processed single-cell table as `.csv` and `.h5ad` (this can be used to reproduce the [Figures]() from the manuscript)

## Processing workflow
**Note:** All paths need to be adjusted to the github repo and locally stored files 
### Image processing with Steinbock
---------------------
---------------------
[Steinbock](https://bodenmillergroup.github.io/steinbock/latest/cli/intro/) has been used with standard parameters:

`steinbock preprocess imc panel`

`steinbock preprocess imc images --hpf 50`

`steinbock segment deepcell --minmax --verbosity INFO`

`steinbock measure intensities`

`steinbock measure regionprops`

`steinbock export anndata --intensities intensities --data regionprops -o cells.h5ad`

### Further preprocessing
---------------------
---------------------
Artifact [removal]() with subsequent [cleaning]() of quantification tables

Bone labeling from [geojson]() using this [script]() and adjusting [quantification]() files accordingly 

### Phenotyping
---------------------
---------------------
For detailed information see our manuscript at:

#### Ground Truth
---------------------
Ground truth was generated with visual manual annotation using scimap's prior knowledge [approach](). **Note:** Scimap version 2.0.5 was used, many functions for hierarchical prior knowlege driven annotation have been updated in more recent versions.

- Single images were [annotated](). Intermediary output files can be found [here]() with one annotated anndata object per image and the stored gates. Aggregating single outputs into csv can be performed [here](). Adjusting existing files to a new hierarchical assignment table can be found in the first part of this [script]()

Phenotyping has been performed with [scanorama]() batch corrected (scanorama_corrected) data, however uncorrected data (uncorrected) is also [present](), therefore data paths in scripts have to be set accordingly. As Scanorama corrected data yielded superior performance in our model, annotations soley rely on scanorama-corrected data.

- Batch correction has been [perforned]() with subsequent [transfer]() to csvs. If uncorrected values should be used, these can be found [here]() (not recommended)
  - For adjusting the per-image anndata annotations from scimap with batch corrected values, refer to the second part of this [script]()
#### Training of XGBoost model
---------------------
- Ground Truth aggregated data was used to train an [xgboost]() model. The script integrates optuna for hyperparameter tuning.
- Classifier was [applied]() on all data. Models are stored [here]() including performances
- Running the model on holdout data was performed [here](). Holdout data annotations can be downloaded form the data repository

### Downstream analysis
---------------------
---------------------
#### Preprocessing
---------------------
- [Transfering]() the classifier annotations (csv format) to the quantification tables from Steinbock output and creating an updated anndata object with comprehensive information
- Further preprocessing steps as removing/relabeling patients can be found [here]()
- Phenotyping refinemed by:
  - [subclustering]() unknown cells to find celltypes missed by xgboost
  - [refinement]() (last part) of macrophages annotations in tumor aggregates. Find the thresholding data needed for this [here]()
- Spatial neighbor graph and Marker normalization can be found [here](). The script further includes some renamings for celltypes and neighborhoods to fit to the mansucript
#### CellCharter
---------------------
- To build the trVAE model and run autoK-Clustering to find the best cluster sizes for cellcharter neighborhoods, refer to [this](). The cellcharter pipeline to extract neighborhood labels can be found [here](). The trVAE model is uploaded in the same directory, autoK models are uploaded to the data repository

#### COZI
---------------------
-s

#### Preprocessing of clinical data and associated spatial scores
---------------------
- Raw Metadata is available as excel sheet on the data repository, to transform into python format and preprocess, use this [script]()
- To get neighborhod enrichment scores per patients and saving them, refer to this [script]()
- To connect COZI scores to patients, refer to this [script]()

#### Downstream analysis of the manuscript
---------------------

All figures can be recreated directly without running the processing workflowe using the anndata object uploaded to the data repository and the interaction scores saved as csv files [here]()
All scripts for creating the figures can be found [here]()
