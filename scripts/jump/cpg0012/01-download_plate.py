#!/usr/bin/env python3
"""
download_plate.py
-----------------
Script to download:
  1) A single plate's images (all channels/sites)
  2) The matching platemap file (e.g., "H-BIOA-004-3.txt")
  3) The load_data.csv for that plate

from the Cell Painting Gallery (cpg0012 dataset), using Quilt3.

Requirements:
    pip install quilt3 pandas
Usage:
    python download_plate.py <plate_id> <barcode_platemap_csv> [<local_out_dir>]

Example:
    python download_plate.py 24277 barcode_platemap.csv .
"""

import sys
import os
import pandas as pd
import quilt3 as q3

# --- Constants pointing to the cpg0012 S3 paths ---
S3_BUCKET = "s3://cellpainting-gallery"
DATASET_PATH = "cpg0012-wawer-bioactivecompoundprofiling"

# The images live here:
IMAGES_PATH = f"{DATASET_PATH}/broad/images/CDRP/images"

# Platemap info:
PLATEMAP_BASE = f"{DATASET_PATH}/broad/workspace/metadata/platemaps/CDRP"
PLATEMAP_SUBFOLDER = f"{PLATEMAP_BASE}/platemap"

# load_data.csv path pattern:
# e.g. "cpg0012-wawer-bioactivecompoundprofiling/broad/workspace/load_data_csv/CDRP/24277/load_data.csv"
LOAD_DATA_BASE = f"{DATASET_PATH}/broad/workspace/load_data_csv/CDRP"


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: download_plate.py <plate_id> <barcode_platemap_csv> [<local_out_dir>]"
        )
        sys.exit(1)

    plate_id = str(sys.argv[1])
    barcode_csv = sys.argv[2]
    local_out_dir = sys.argv[3] if len(sys.argv) > 3 else "."

    # 1) Load the barcode_platemap.csv (which you have locally)
    df_barcode = pd.read_csv(barcode_csv)
    # Expect columns: Plate_Map_Name, Assay_Plate_Barcode (like "24,277")

    df_barcode["Assay_Plate_Barcode_str"] = (
        df_barcode["Assay_Plate_Barcode"].astype(str).str.replace(",", "")
    )

    # 2) Find the row for this plate_id
    row = df_barcode.loc[df_barcode["Assay_Plate_Barcode_str"] == plate_id]
    if row.empty:
        print(f"Error: Plate ID {plate_id} not found in {barcode_csv}.")
        sys.exit(1)

    platemap_name = row["Plate_Map_Name"].iloc[0]
    print(f"Plate ID {plate_id} → Platemap: {platemap_name}")

    # 3) Use Quilt to download images
    bucket = q3.Bucket(S3_BUCKET)

    plate_image_path = f"{IMAGES_PATH}/{plate_id}/"
    local_plate_dir = os.path.join(local_out_dir, plate_id)
    if not local_plate_dir.endswith("/"):
        local_plate_dir += "/"
    print(f"Downloading images from {plate_image_path} to {local_plate_dir} ...")
    bucket.fetch(plate_image_path, local_plate_dir)

    # 4) Download the matching platemap file
    platemap_file_remote = f"{PLATEMAP_SUBFOLDER}/{platemap_name}.txt"
    platemap_file_local = os.path.join(local_out_dir, f"{platemap_name}.txt")
    print(
        f"Downloading platemap from {platemap_file_remote} to {platemap_file_local} ..."
    )
    bucket.fetch(platemap_file_remote, platemap_file_local)

    # 5) Download the load_data.csv for this plate (if it exists)
    # cpg0012-wawer-bioactivecompoundprofiling/broad/workspace/load_data_csv/CDRP/<plate_id>/load_data.csv
    load_data_remote = f"{LOAD_DATA_BASE}/{plate_id}/load_data.csv"
    load_data_local = os.path.join(local_out_dir, f"load_data_{plate_id}.csv")
    print(
        f"Attempting to download load_data.csv from {load_data_remote} to {load_data_local} ..."
    )

    # If there's no load_data.csv for this plate, fetch() may raise an error; you can catch it:
    try:
        bucket.fetch(load_data_remote, load_data_local)
    except Exception as e:
        print(f"Warning: Could not download load_data.csv for plate {plate_id}.\n{e}")

    print("Done.")


if __name__ == "__main__":
    main()
