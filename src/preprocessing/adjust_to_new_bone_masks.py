import numpy as np
import pandas as pd
import os
import tifffile as tiff
from scipy.ndimage import distance_transform_edt

print("succesfully loaded packages")

def get_distance(row, distance_from_bone):
    if np.isnan(distance_from_bone).all():
        return np.nan
    x, y = round(row['centroid-1']), round(row['centroid-0'])
    y = max(0, min(y, distance_from_bone.shape[0] - 1))
    x = max(0, min(x, distance_from_bone.shape[1] - 1))
    return distance_from_bone[y, x]

# Directory containing the bone mask multichannel TIFFs
mask_dir = "/Users/lukashat/Documents/PhD_Schapiro/Projects/Myeloma_Standal/results/artifact_bone_masks/bone_masks/"

# Directory containing the newly generated merged CSV tables
merged_dir = "/Users/lukashat/Documents/PhD_Schapiro/Projects/Myeloma_Standal/results/artifact_bone_masks/merged_csv"
regionprops_dir = "/Users/lukashat/Documents/PhD_Schapiro/Projects/Myeloma_Standal/results/artifact_bone_masks/regionprops"

for element in os.listdir(mask_dir):
    if element.endswith("_bone_masks.tiff"):
        try:
            # Read the bone mask multichannel TIFF
            mask_image = tiff.imread(os.path.join(mask_dir, element))
            
            # Check the shape of the bone mask and set layer indices accordingly
            if mask_image.shape == (1000, 1000):
                print('correct shape')
            else:
                raise ValueError(f"Unexpected shape of bone mask: {element}")
            
            if np.all(mask_image == 0):
                print(f"Empty mask detected: {element}")
                distance_from_bone = np.full_like(mask_image, np.nan, dtype=float)
            else:
                distance_from_bone = distance_transform_edt(mask_image == 0)

            # Get the associated merged and regionprops CSV tables
            base_filename = os.path.splitext(element)[0].replace("_bone_masks", "")
            merged_csv = os.path.join(merged_dir, base_filename + ".csv")
            regionprops_csv = os.path.join(regionprops_dir, base_filename + ".csv")
            
            # Update the merged CSV
            merged_data = pd.read_csv(merged_csv)
            merged_data['distance_to_bone_corrected'] = merged_data.apply(lambda row: get_distance(row, distance_from_bone), axis=1)
            merged_data.to_csv(merged_csv, index=False)
            
            # Update the regionprops CSV
            regionprops_data = pd.read_csv(regionprops_csv)
            regionprops_data['distance_to_bone_corrected'] = regionprops_data.apply(lambda row: get_distance(row, distance_from_bone), axis=1)
            regionprops_data.to_csv(regionprops_csv, index=False)
            
        except Exception as e:
            print(f"Error processing file {element}: {e}")