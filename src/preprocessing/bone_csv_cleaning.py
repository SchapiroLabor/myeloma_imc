import pandas as pd
import numpy as np
import argparse
import os


def clean_bone(base_regionprops_path, base_merged_path, files, output_path):
    output_regionprops_path = os.path.join(output_path, 'regionprops')
    output_merged_path = os.path.join(output_path, 'merged_csv')
    for path in [output_regionprops_path, output_merged_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    files = files.split(',')

    for file in files:
        file_regionprops_path = os.path.join(base_regionprops_path, f"{file}.csv")
        file_merged_path = os.path.join(base_merged_path, f"{file}.csv")

        if os.path.exists(file_regionprops_path) and os.path.exists(file_merged_path):
            print(f"Processing {file}...")
            try:
                regionprops = pd.read_csv(file_regionprops_path)
                merged = pd.read_csv(file_merged_path)

                regionprops["near_bone"] = False
                regionprops["distance_to_bone"] = np.nan
                merged["near_bone"] = False
                merged["distance_to_bone"] = np.nan

                regionprops.to_csv(os.path.join(output_regionprops_path, f"{file}.csv"), index=False)
                merged.to_csv(os.path.join(output_merged_path, f"{file}.csv"), index=False)
                print(f"Files {file_regionprops_path} and {file_merged_path} have been cleaned and stored in {output_regionprops_path} and {output_merged_path} respectively.")
            except Exception as e:
                print(f"Error processing {file}: {e}")
        else:
            if not os.path.exists(file_regionprops_path):
                print(f"Regionprops file {file_regionprops_path} does not exist.")
            if not os.path.exists(file_merged_path):
                print(f"Merged file {file_merged_path} does not exist.")
    


def main():
    parser = argparse.ArgumentParser(description="Remove bone labels if wrong bone masks have been created due to no or artificial staining.")
    parser.add_argument("-r", "--regionprops_path", required=True, help="Path to the folder containing regionprops tables")
    parser.add_argument("-m", "--merged_path", required=True, help="Path to the folder containing merged csv tables")
    parser.add_argument("-f", "--files", required=True, help="A list of files to be cleaned. The files should be separated by a comma.")
    parser.add_argument("-o", "--output_path", required=True, help="Path to the output directory for cleaned files. It should contain 3 subdirectories named intensities, regionprops and merged_csv")

    args = parser.parse_args()
    clean_bone(args.regionprops_path, args.merged_path, args.files, args.output_path)

if __name__ == "__main__":
    main()