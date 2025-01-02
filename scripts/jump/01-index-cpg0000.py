import re
import asyncio
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import pandas as pd
import aioboto3

# Initialize S3 client without credentials (for public bucket access)
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

# source: https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods
CHANNEL_NAMES = [
    "ch01",  # Alexa 647 -> MITO?
    "ch02",  # Alexa 568 -> AGP?
    "ch03",  # Alexa 488 (long) -> RNA?
    "ch04",  # Alexa 488 -> ER?
    "ch05",  # Hoechst 33342 -> DNA?
    "ch06",  # brightfield plane 1
    "ch07",  # brightfield plane 2
    "ch08",  # brightfield plane 3
]

# Just for making the debugger work
SCRIPT_DIR = Path(__file__).parent


async def list_prefix_contents_async(prefix: str, session) -> list:
    """Async version of listing prefix contents"""
    folders = []
    async with session.client(
        "s3", config=Config(signature_version=UNSIGNED)
    ) as s3_async:
        paginator = s3_async.get_paginator("list_objects_v2")
        async for page in paginator.paginate(
            Bucket="cellpainting-gallery", Delimiter="/", Prefix=prefix
        ):
            if "CommonPrefixes" in page:
                for prefix_obj in page["CommonPrefixes"]:
                    folders.append(prefix_obj["Prefix"])
    return folders


def list_prefix_contents(prefix: str) -> list:
    """List contents of a specific prefix in the bucket"""
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(
        Bucket="cellpainting-gallery", Delimiter="/", Prefix=prefix
    )

    folders = []
    for page in pages:
        if "CommonPrefixes" in page:
            for prefix_obj in page["CommonPrefixes"]:
                folders.append(prefix_obj["Prefix"])
    return folders


def get_plate_id_from_path(path: str) -> str:
    """
    Extract plate ID based on the pattern for source_4

    Args:
        path (str): Full S3 path

    Returns:
        str: Extracted plate ID or None if no match
    """
    # Pattern: BR00117035__2021-05-02T16_02_51-Measurement1
    match = re.search(r"(BR\d+)__\d{4}", path)
    return match.group(1) if match else None


async def list_images_in_plate_async(plate_path: str, session) -> list:
    """
    Async version of listing images in a plate

    Args:
        plate_path (str): Base path to plate
        session: aioboto3 session
    """
    # For source_4, images are in 'Images' subfolder
    image_path = plate_path.rstrip("/") + "/Images/"
    image_files = []

    async with session.client(
        "s3", config=Config(signature_version=UNSIGNED)
    ) as s3_async:
        paginator = s3_async.get_paginator("list_objects_v2")
        async for page in paginator.paginate(
            Bucket="cellpainting-gallery", Prefix=image_path
        ):
            if "Contents" in page:
                for obj in page["Contents"]:
                    if obj["Key"].endswith(".tiff"):
                        image_files.append(obj["Key"])

    return image_files


def get_well_info_from_filename(filename: str) -> tuple:
    """
    Extract well information from filename based on source_4 patterns

    Returns:
        tuple: (row, col, field, channel) or None if no match
    """
    # Pattern: r01c02f01p01-ch1sk1fk1fl1.tiff
    match = re.match(r"r(\d+)c(\d+)f(\d+)p\d+-ch(\d+)", filename)
    if match:
        return match.groups()
    else:
        return None


def organize_images_by_well_field(images: list) -> dict:
    """
    Organize images by well and field combination

    Returns a structure like:
    {
        ("A01", "1"): {
            "channels": {1: path1, 2: path2, ...}
        },
        ("A01", "2"): {
            "channels": {1: path1, 2: path2, ...}
        }
    }
    """
    well_field_images = {}

    for img in images:
        filename = img.split("/")[-1]
        well_info = get_well_info_from_filename(filename)
        if well_info:
            row, col, field, channel = well_info

            # Convert row number to letter and ensure column is zero-padded
            row_letter = chr(int(row) + ord("A") - 1)
            well = f"{row_letter}{int(col):02d}"

            # Use (well, field) tuple as key
            well_field_key = (well, field)
            if well_field_key not in well_field_images:
                well_field_images[well_field_key] = {
                    "channels": {i: None for i in range(1, 9)},
                }

            cnum = int(channel)
            if 1 <= cnum <= 8:
                well_field_images[well_field_key]["channels"][cnum] = img

    return well_field_images


import re


def get_illum_prefix(plate_path: str) -> str:
    """
    Construct the S3 prefix for illumination files from the images prefix,
    removing the __<date/time>-Measurement suffix from the plate folder name.

    Example:
      If plate_path =
        ".../images/2020_11_04_CPJUMP1/images/BR00117017__2020-11-10T18_25_46-Measurement1/"
      Then illum prefix =
        ".../images/2020_11_04_CPJUMP1/illum/BR00117017/"
    """
    parent_path = plate_path.rstrip("/")
    plate_name_with_suffix = parent_path.split("/")[
        -1
    ]  # e.g. "BR00117017__2020-11-10T18_25_46-Measurement1"

    # Use regex to remove "__..." and everything after it
    # So "BR00117017__2020-11-10T18_25_46-Measurement1" → "BR00117017"
    base_plate_name = re.sub(r"__.*", "", plate_name_with_suffix)

    # Step up one level (remove the plate folder itself)
    parent_dir = parent_path.rsplit("/", 1)[0]
    # Step up one more level (remove 'images' folder)
    grandparent = parent_dir.rsplit("/", 1)[0]

    # Now build the illum prefix:
    # ".../illum/BR00117017/"
    illum_prefix = f"{grandparent}/illum/{base_plate_name}/"
    return illum_prefix


async def list_illum_files_for_plate_async(plate_path: str, session) -> dict:
    """
    List all illumination .npy files for a plate and return
    a dictionary mapping channel_name -> S3 key.
    e.g. {
      "AGP": "cpg0000-jump-.../BR00116991_IllumAGP.npy",
      "DNA": "...",
      ...
    }
    """
    illum_dict = {}
    illum_prefix = get_illum_prefix(plate_path)

    print(f"Listing illumination files for {plate_path}...")
    print(f"Illumination prefix: {illum_prefix}")

    async with session.client(
        "s3", config=Config(signature_version=UNSIGNED)
    ) as s3_async:
        paginator = s3_async.get_paginator("list_objects_v2")
        async for page in paginator.paginate(
            Bucket="cellpainting-gallery", Prefix=illum_prefix
        ):
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    if key.endswith(".npy"):
                        # Example filename: BR00116991_IllumAGP.npy
                        filename = key.split("/")[-1]
                        # Parse channel name out of "BR00116991_IllumAGP.npy"
                        # Everything after "Illum" and before ".npy"
                        match = re.search(r"_Illum(.*)\.npy", filename)
                        if match:
                            channel_name = match.group(1)
                            illum_dict[channel_name] = key

    print(f"Found illumination files: {illum_dict} for {plate_path}")
    return illum_dict


async def process_plate(plate_path: str, session):
    images = await list_images_in_plate_async(plate_path, session)
    well_field_data = organize_images_by_well_field(images)
    print(f"\nProcessing plate: {plate_path}")
    print(f"(well, field) combos in DataFrame: {len(well_field_data.keys())}")

    # Get the illumination .npy file paths
    illum_map = await list_illum_files_for_plate_async(plate_path, session)

    # Convert well_field_data to DataFrame rows
    rows = []
    for (well, field), data in well_field_data.items():
        row_data = {
            "Metadata_Plate": plate_path.split("/")[-2],  # or parse as you like
            "Metadata_Well": well,
            "Metadata_Field": int(field),
        }
        # Add image paths
        for channel_num, path in data["channels"].items():
            row_data[f"Image_{CHANNEL_NAMES[channel_num - 1]}"] = path

        # Add illumination paths
        # e.g. row_data["Illum_AGP"] = illum_map.get("AGP", None)
        # We'll add a column for each illum channel actually found:
        for channel_name, npy_path in illum_map.items():
            # Create a column like Illum_AGP, Illum_DNA, etc.
            row_data[f"Illum_{channel_name}"] = npy_path

        rows.append(row_data)

    return rows


async def main_async():
    project_name = "cpg0000-jump-pilot"
    source = "source_4"
    output_file = SCRIPT_DIR / f"data/{project_name}-{source}.parquet"

    print(f"Listing plates for {project_name}/{source}...")

    # Get plate paths
    batch_prefix = f"{project_name}/{source}/images/"
    batches = list_prefix_contents(batch_prefix)

    plate_paths = []
    for batch in batches:
        image_prefix = f"{batch}images/"
        plates = list_prefix_contents(image_prefix)
        for plate_path in plates:
            plate_id = get_plate_id_from_path(plate_path)
            if plate_id:
                plate_paths.append(plate_path)

    print(f"Found {len(plate_paths)} plates.")

    # Create a session for reuse
    session = aioboto3.Session()

    # Process plates in parallel
    tasks = [process_plate(plate_path, session) for plate_path in plate_paths]
    results = await asyncio.gather(*tasks)

    # Flatten the list of lists
    all_rows = [row for plate_rows in results for row in plate_rows]

    # Create DataFrame
    result_df = pd.DataFrame(all_rows).reset_index(drop=True)
    print(f"\nCreated DataFrame with {len(result_df)} rows (well-field combinations)")

    # Display results
    print("\nFinal DataFrame columns:")
    print(result_df.columns.tolist())

    print("\nSample row:")
    # Show a subset of columns (well + a couple image channels + a couple illum columns)
    display_cols = (
        ["Metadata_Well"]
        + [f"Image_{ch}" for ch in CHANNEL_NAMES[:2]]
        + [col for col in result_df.columns if col.startswith("Illum_")][:2]
    )  # Just show first 2 Illum columns
    print(result_df.iloc[0][display_cols])

    # Check for missing images
    print("\nMissing images per channel:")
    for channel in CHANNEL_NAMES:
        missing = result_df[f"Image_{channel}"].isna().sum()
        print(f"{channel}: {missing} missing")

    # Check for missing illum
    illum_cols = [c for c in result_df.columns if c.startswith("Illum_")]
    print("\nMissing illumination files per channel type:")
    for illum_col in illum_cols:
        missing = result_df[illum_col].isna().sum()
        print(f"{illum_col}: {missing} missing")

    # Save the DataFrame
    print(f"\nSaving DataFrame to {output_file}")
    result_df.to_parquet(output_file)
    print("Done!")

    # Print some stats about the saved data
    print(f"\nDataset statistics:")
    print(f"Total rows (well-fields): {len(result_df)}")
    print(f"Total plates: {len(result_df['Metadata_Plate'].unique())}")


if __name__ == "__main__":
    asyncio.run(main_async())
