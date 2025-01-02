#!/usr/bin/env python

import argparse
import asyncio
import io
from pathlib import Path

import aioboto3
import numpy as np
import pandas as pd
import tifffile
import torch
import webdataset as wds
from botocore import UNSIGNED
from botocore.config import Config
from sklearn.model_selection import train_test_split
from torchvision.transforms.functional import resize
from tqdm import tqdm


# -------------------------
# 1. STRATIFIED SAMPLING
# -------------------------
def stratified_sample(df, strata_cols, frac, random_state=None):
    """
    Perform a stratified sample of the DataFrame based on the specified strata columns.

    Args:
        df (pd.DataFrame): The DataFrame to sample from.
        strata_cols (list): The columns to use for stratification.
        frac (float): Fraction of the dataset to sample (0 < frac <= 1).
        random_state (int, optional): A seed for the random number generator.

    Returns:
        pd.DataFrame: The stratified sampled DataFrame.
    """
    grouped = df.groupby(strata_cols, group_keys=False)
    sampled_df = grouped.apply(lambda x: x.sample(frac=frac, random_state=random_state))
    return sampled_df.reset_index(drop=True)


# -------------------------
# 2. ASYNC FUNCTIONS
# -------------------------
async def download_sample_async(session, row, tar):
    """
    Downloads 5 fluorescent channels from S3, applies illumination correction
    (via division), resizes, and then writes them to 'tar' as a single NPY file.
    Raises an error if the illumination function is missing for any channel.
    """

    channels_corrected = []
    async with session.client("s3", config=Config(signature_version=UNSIGNED)) as s3:

        for channel in ["RNA", "ER", "AGP", "Mito", "DNA"]:
            # (1) Get raw image path
            s3_path_img = row.get(f"Image_{channel}", None)
            if not s3_path_img or pd.isna(s3_path_img):
                raise ValueError(
                    f"Raw image missing for channel '{channel}' at row {row.name}."
                )

            # (2) Get illumination file path
            s3_path_illum = row.get(f"Illum_{channel}", None)
            if not s3_path_illum or pd.isna(s3_path_illum):
                # Raise an error if the illum function doesn't exist
                raise ValueError(
                    f"No illumination function found for channel '{channel}' at row {row.name}.\n"
                    f"Raw image path = {s3_path_img}"
                )

            # (3) Download raw image
            response_img = await s3.get_object(
                Bucket="cellpainting-gallery", Key=s3_path_img
            )
            img_data = await response_img["Body"].read()
            raw_img = tifffile.imread(io.BytesIO(img_data))

            # (4) Download illum function
            response_illum = await s3.get_object(
                Bucket="cellpainting-gallery", Key=s3_path_illum
            )
            illum_data = await response_illum["Body"].read()
            illum = np.load(io.BytesIO(illum_data))

            # (5) Apply correction: raw_img / illum
            epsilon = 1e-8  # Avoid divide-by-zero
            corrected_img = raw_img / (illum + epsilon)

            channels_corrected.append(corrected_img)

    # Now we have a list of corrected 2D arrays: shape list [ (H,W), ... ].
    if len(channels_corrected) == 0:
        # If we somehow had no valid channels, skip writing (or raise an error)
        raise ValueError(f"No channels processed at row {row.name}.")

    # Stack into 5-channel array, shape = (C,H,W)
    combined = np.stack(channels_corrected, axis=0).astype(np.float32)

    # Resize the stacked array (all channels together)
    combined_tensor = torch.from_numpy(combined)
    resized_tensor = resize(combined_tensor, [512, 512])  # (C,512,512)

    # Save to buffer
    buffer = io.BytesIO()
    np.save(buffer, resized_tensor.numpy())
    buffer.seek(0)

    # Write to tar
    tar.write(
        {
            "__key__": f"sample_{row.name:08d}",
            "image.npy": buffer.getvalue(),
        }
    )


async def create_tar_shard_async(session, shard_df, tar_path):
    """
    Process one tar shard. Downloads images in parallel for each row in shard_df,
    then writes them into a .tar file via WebDataset.
    """
    with wds.TarWriter(str(tar_path)) as tar:
        tasks = [
            download_sample_async(session, row, tar) for _, row in shard_df.iterrows()
        ]
        await asyncio.gather(*tasks)


async def download_and_create_tar_async(
    split_df, output_dir, split_name, samples_per_tar=100
):
    """
    Downloads images for a given split_df and creates tar files in chunks.
    """
    output_dir = Path(output_dir) / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    session = aioboto3.Session()

    # Process in chunks for tar files
    tasks = []
    for shard_idx in range(0, len(split_df), samples_per_tar):
        shard_df = split_df.iloc[shard_idx : shard_idx + samples_per_tar]
        tar_path = output_dir / f"shard_{shard_idx:08d}.tar"

        task = create_tar_shard_async(session, shard_df, tar_path)
        tasks.append(task)

    # Run all shards in parallel with progress bar
    for coro in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"Processing {split_name} shards",
    ):
        await coro


# -------------------------
# 3. MAIN ENTRY POINT (CLI)
# -------------------------
async def main_async(args):
    """
    1) Load full dataframe from args.df_path
    2) Stratify sample to get 'args.subset' fraction
    3) Train/val/test split
    4) WebDataset creation
    """
    # 3.1 Load DataFrame
    df = pd.read_parquet(args.df_path)
    print(f"Full DataFrame loaded: {len(df):,} rows.")

    # 3.2 Stratified Sample
    if args.subset < 1.0:
        print(f"Performing stratified sampling at fraction = {args.subset}")
        df = stratified_sample(
            df, args.strata_cols, frac=args.subset, random_state=args.random_state
        )
        print(f"Subset DataFrame has {len(df):,} rows after stratification.")
        subset_str = str(args.subset)
        output_dir = f"data/jump_{subset_str}"
    else:
        print("Using the full dataset (no sampling).")
        output_dir = "data/jump_full"

    # 3.3 Train/Val/Test split
    train_val_df, test_df = train_test_split(
        df, test_size=0.1, random_state=args.random_state
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.1, random_state=args.random_state
    )

    print(f"Final splits:")
    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    # 3.4 Create .tar shards for each split
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for split_name, split_df in splits.items():
        await download_and_create_tar_async(
            split_df, output_dir, split_name, samples_per_tar=args.samples_per_tar
        )


def main():
    parser = argparse.ArgumentParser(
        description="Download images for Jump dataset with illumination correction."
    )
    parser.add_argument(
        "--df_path",
        type=str,
        default="scripts/jump/data/cpg0000_dataframe.parquet",
        help="Path to the full DataFrame parquet file.",
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=0.001,
        help="Fraction of data to stratify-sample (e.g. 0.001 = 0.1%). If >=1.0, use full dataset.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for stratified sampling and train/test splitting.",
    )
    parser.add_argument(
        "--strata_cols",
        nargs="+",
        default=["Metadata_PlateType", "Metadata_Source"],
        help="Columns to use for stratified sampling.",
    )
    parser.add_argument(
        "--samples_per_tar",
        type=int,
        default=10,
        help="Number of samples to store per .tar shard.",
    )

    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
