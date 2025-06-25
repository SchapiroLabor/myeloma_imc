# myeloma_imc
Repo for reproducing the spatial analysis on the myeloma cohort from NTNU (Standal Group) for the manuscript: [TBD]

![Visual abstract](./Visual%20abstract.png)

## Raw data
Raw data will be uploaded to:

Raw data includes:

- Images, segmentation masks, bone masks, steinbock raw outputs, patient metadata
- Intermediary files for phenotyping (e.g. scimap gates, scimap outputs, manually annotated and aggregated batch-corrected intensities CSVs, XGB models including performances and parameters, holdout test data for xgboost) and other models like cellcharter AutoK and trVAE models
- Processed single-cell table as `.csv` and `.h5ad` (this can be used to reproduce the [Figures](https://github.com/SchapiroLabor/myeloma_imc/tree/main/src/Paper/figure_plots) from the manuscript)

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
Artifact [removal](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/preprocessing/generate_artifact_coordinates.ipynb) with subsequent [cleaning](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/preprocessing/clean_quantification_tables.py) of quantification tables

Bone labeling from [geojson]() using this [script](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/preprocessing/extract_bone_masks_geojson.ipynb) and adjusting [quantification](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/preprocessing/adjust_to_new_bone_masks.py) files accordingly 

### Phenotyping
---------------------
---------------------
For detailed information on the different steps taken see our manuscript at: [TBD]

#### Ground Truth
---------------------
Ground truth was generated with visual manual annotation using scimap's prior knowledge [approach](https://scimap.xyz/tutorials/md/scimap_phenotyping/). **Note:** Scimap version 2.0.5 was used, many functions for hierarchical prior knowlege driven annotation have been updated in more recent versions.

- Single images were [annotated](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/Phenotyping/scimap/scimap_phenotyping_batch.ipynb). Intermediary output files can be found [here](https://github.com/SchapiroLabor/myeloma_imc/tree/main/phenotyping/uncorrected/standard) with one annotated anndata object per image and the stored gates. Aggregating single outputs into csv can be performed [here](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/Phenotyping/preprocess/combine_adata_phenotypes.ipynb). Adjusting existing files to a new hierarchical assignment table can be found in the first part of this [script](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/Phenotyping/preprocess/adjusting_adata_to_scheme.ipynb)

Phenotyping has been performed with [scanorama](https://github.com/brianhie/scanorama) batch corrected (scanorama_corrected) data, however uncorrected data (uncorrected) is also [present](https://github.com/SchapiroLabor/myeloma_imc/tree/main/phenotyping/uncorrected), therefore data paths in scripts have to be set accordingly. As Scanorama corrected data yielded superior performance in our model, annotations soley rely on scanorama-corrected data.

- Batch correction has been [perforned](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/Phenotyping/preprocess/scanorama_correction.py) with subsequent [transfer](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/Phenotyping/preprocess/batch_correction_transfer_to_csv.ipynb) to csvs. If uncorrected values should be used, these can be found in the data repository (not recommended)
  - For adjusting the per-image anndata annotations from scimap with batch corrected values, refer to the second part of this [script](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/Phenotyping/preprocess/adjusting_adata_to_scheme.ipynb)
#### Training of XGBoost model
---------------------
- Ground Truth aggregated data was used to train an [xgboost](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/Phenotyping/xgboost/xgboost_optuna.py) model. The script integrates optuna for hyperparameter tuning.
- Classifier was [applied](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/Phenotyping/applying_classifiers/apply_classifier_xgboost_comprehensive.ipynb) on all data. Models are uploaded to the data repository (see above)
- Running the model on holdout data was performed [here](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/Phenotyping/applying_classifiers/apply_classifier_holdout_test.ipynb). Holdout data annotations can be downloaded form the data repository

### Downstream analysis
---------------------
---------------------
#### Preprocessing
---------------------
- [Transfering](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/downstream/preprocessing/annotation_downstream.ipynb) the classifier annotations (csv format) to the quantification tables from Steinbock output and creating an updated anndata object with comprehensive information
- Further preprocessing steps as removing/relabeling patients can be found [here](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/downstream/preprocessing/preprocess.ipynb)
- Phenotyping refinemed by:
  - [subclustering](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/downstream/preprocessing/unknown_subclustering.ipynb) unknown cells to find celltypes missed by xgboost
  - [refinement](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/downstream/preprocessing/preprocess.ipynb) (last part) of macrophages annotations in tumor aggregates. Find the thresholding data needed for this in the public data repository and [here](https://github.com/SchapiroLabor/myeloma_imc/tree/main/src/downstream/preprocessing/thresholding)
- Spatial neighbor graph and Marker normalization can be found [here](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/downstream/preprocessing/marker_normalization_preprocessing.ipynb). The script further includes some renamings for celltypes and neighborhoods to fit to the mansucript
#### CellCharter
---------------------
- To build the trVAE model and run autoK-Clustering to find the best cluster sizes for cellcharter neighborhoods, refer to [this](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/downstream/cellcharter/cellcharter_autok.ipynb). The cellcharter pipeline to extract neighborhood labels can be found [here](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/downstream/cellcharter/cellcharter.ipynb). The trVAE model is uploaded in the same directory, autoK models are uploaded to the data repository

#### COZI
---------------------
Running Cozi to infer NEP scores and saving results can be found [here](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/downstream/COZI/cozi_MM.ipynb)

#### Preprocessing of clinical data and associated spatial scores
---------------------
- Raw Metadata is available as excel sheet on the data repository, to transform into python format and preprocess, use this [script](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/downstream/clinical_correlation/integrate_metadata.ipynb)
- To get neighborhod enrichment scores per patients and saving them, refer to this [script](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/downstream/clinical_correlation/cellcharter_long_short_pfs.ipynb)
- To connect COZI scores to patients, refer to this [script](https://github.com/SchapiroLabor/myeloma_imc/blob/main/src/downstream/clinical_correlation/immune_clinical.ipynb)

#### Downstream analysis of the manuscript
---------------------

All figures can be recreated directly without running the processing workflowe using the anndata object uploaded to the data repository and the interaction scores saved as csv files [here](https://github.com/SchapiroLabor/myeloma_imc/tree/main/src/downstream/clinical_correlation)
All scripts for creating the figures can be found [here](https://github.com/SchapiroLabor/myeloma_imc/tree/main/src/Paper/figure_plots)
