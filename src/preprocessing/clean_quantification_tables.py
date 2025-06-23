import pandas as pd
from shapely.geometry import Point, Polygon
import ast
import argparse
import os
import numpy as np
import re

def parse_polygon(coord_str):
    try:
        coord_str = coord_str.strip("[]")
        coord_pairs = re.findall(r'\(.*?\)', coord_str)
        coord_list = [ast.literal_eval(pair) for pair in coord_pairs]

        # Flatten the list of coordinates
        coord_list = [item for sublist in coord_list for item in sublist]

        # Create a Polygon object
        polygon = Polygon(coord_list)
        return polygon
    except Exception as e:
        print(f"Error parsing polygon coordinates: {coord_str}")
        raise e

def is_within_polygon(y, x, polygons):
    point = Point(y, x)
    return any(polygon.contains(point) for polygon in polygons)


def process_files(intensities_path, regionprops_path, artifacts_path, output_path, merged_path):
    artifacts = pd.read_csv(artifacts_path)
    artifacts['polygon'] = artifacts['coordinates'].apply(parse_polygon)

    output_regionprops_path = os.path.join(output_path, 'regionprops')
    output_intensities_path = os.path.join(output_path, 'intensities')
    output_merged_path = os.path.join(output_path, 'merged_csv')
    for path in [output_regionprops_path, output_intensities_path, output_merged_path, merged_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    # Process each artifact entry
    for _, artifact in artifacts.iterrows():
        image_name = artifact['image_file'].replace('.tiff', '')
        intensities_file = os.path.join(intensities_path, f"{image_name}.csv")
        regionprops_file = os.path.join(regionprops_path, f"{image_name}.csv")
        merged_file = os.path.join(merged_path, f"{image_name}.csv")

        if os.path.exists(intensities_file) and os.path.exists(regionprops_file):
            print(f"Processing {image_name}...")
            intensities = pd.read_csv(intensities_file)
            regionprops = pd.read_csv(regionprops_file)
            merged = pd.read_csv(merged_file)

            # Check each cell
            regionprops['is_artifact'] = regionprops.apply(
                lambda row: is_within_polygon(row['centroid-0'], row['centroid-1'], [artifact['polygon']]), axis=1)

            # Filter out artifacts
            clean_regionprops = regionprops[regionprops['is_artifact'] == False]
            clean_intensities = intensities[intensities['Object'].isin(clean_regionprops['Object'])]
            clean_merged = merged[merged['Object'].isin(clean_regionprops['Object'])]

            # Save cleaned data
            clean_regionprops.to_csv(os.path.join(output_regionprops_path, f"{image_name}.csv"), index=False)
            clean_intensities.to_csv(os.path.join(output_intensities_path, f"{image_name}.csv"), index=False)
            clean_merged.to_csv(os.path.join(output_merged_path, f"{image_name}.csv"), index=False)
            print(f"Saved cleaned files for {image_name} in {output_path}.")
        else:
            print(f"Files for {image_name} not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove artifacts from quantification tables after Steinbock pipeline of IMC data based on polygon coordinates.")
    parser.add_argument("-i", "--intensities_path", required=True, help="Path to the folder containing intensities tables")
    parser.add_argument("-r", "--regionprops_path", required=True, help="Path to the folder containing regionprops tables")
    parser.add_argument("-m", "--merged_path", required=True, help="Path to the folder containing merged csv tables")
    parser.add_argument("-a", "--artifacts_path", required=True, help="Path to the artifacts CSV file. Needs to have a Colum with specified image names and one with polygon coordinates")
    parser.add_argument("-o", "--output_path", required=True, help="Path to the output directory for cleaned files. It should contain 3 subdirectories named intensities, regionprops and merged_csv")

    args = parser.parse_args()

    process_files(args.intensities_path, args.regionprops_path, args.artifacts_path, args.output_path, args.merged_path)