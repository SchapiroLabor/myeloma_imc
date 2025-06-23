import tifffile as tiff
import numpy as np
import os
import re

def extract_channel_number(filename):
    # This regular expression finds numbers after 'channel_' and before the first underscore '_'
    match = re.search(r'channel_(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1

directory = "/gpfs/bwfor/work/ws/hd_bm327-myeloma_standal_ws/IMC42_denoised/combined_channels"
tiff_files = [file for file in os.listdir(directory) if file.endswith(".tiff")]
tiff_files.sort(key=extract_channel_number)

tiff_data = []

for file in tiff_files:
    tiff_data.append(tiff.imread(os.path.join(directory, file)))

multichannel_tiff = np.stack(tiff_data, axis=0)

tiff.imwrite("/gpfs/bwfor/work/ws/hd_bm327-myeloma_standal_ws/IMC42_denoised/img/TS-373_IMC42_B_002.tiff", multichannel_tiff)