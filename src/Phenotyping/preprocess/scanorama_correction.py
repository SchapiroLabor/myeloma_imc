import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import scanorama

sc._settings.ScanpyConfig.n_jobs = -1

adata = ad.read_h5ad('/Users/lukashat/Documents/PhD_Schapiro/Projects/Myeloma_Standal/QC/standard/cells.h5ad')

adata.obs['patient_id'] = adata.obs['image'].str.split('_').str[1]

vars_to_remove = [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', '191Ir', '193Ir', 'HistoneH3']
adata = adata[:, ~adata.var_names.isin(vars_to_remove)]


adata.raw = adata
adata.X = np.arcsinh(adata.X / 1)

obs_to_remove = ['image_num_channels', 'image_recovery_file', 'image_source_file', 'image_recovered','image_acquisition_id','image_acquisition_start_x_um','image_acquisition_start_y_um','image_acquisition_end_x_um', 'image_acquisition_end_y_um', 'image_acquisition_width_um', 'image_acquisition_height_um']
adata.obs = adata.obs.drop(columns=obs_to_remove)

print(adata.var_names)

# scanorama preprocess
sc.pp.neighbors(adata, n_neighbors=10, key_added='neighbors_raw')
sc.pp.pca(adata, n_comps=10)
sc.tl.umap(adata, neighbors_key='neighbors_raw')

adatas_split = []
for patient in adata.obs['patient_id'].unique():
    adatas_split.append(adata[adata.obs['patient_id'] == patient].copy())

#Processing with scanorama
adatas_cor = scanorama.correct_scanpy(adatas_split, return_dimred=True)
print(adatas_cor.shape)

adatas_corrrected = ad.concat(adatas_cor)
print(adatas_corrected)

adatas_corrrected.write_h5ad('/Users/lukashat/Documents/PhD_Schapiro/Projects/Myeloma_Standal/QC/standard/cells_scanorama.h5ad')
print('Scanorama correction done')
