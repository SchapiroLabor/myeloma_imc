import pandas as pd
import os
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--intensities", type=str, help="Path to the intensities directory")
parser.add_argument("-r", "--region", type=str, help="Path to the regionprops directory")
parser.add_argument("-o", "--output", type=str, default='merged_csv', help="Path to the output directory")
args = parser.parse_args()

intensities_dir = args.intensities
regionprops_dir = args.region
output_dir = args.output

os.makedirs(output_dir, exist_ok=True)
intensities_csv = os.listdir(intensities_dir)

for file in intensities_csv:
    intensities_file_path = os.path.join(intensities_dir, file)
    regionprops_file_path = os.path.join(regionprops_dir, file)
    if os.path.exists(regionprops_file_path):
        df_intensities = pd.read_csv(intensities_file_path)
        df_regionprops = pd.read_csv(regionprops_file_path)
        merged_df = pd.merge(df_intensities, df_regionprops, on='Object')
        merged_df.to_csv(os.path.join(output_dir,file), index=False)