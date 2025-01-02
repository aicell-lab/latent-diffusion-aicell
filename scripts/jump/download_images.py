import asyncio
import aioboto3
from botocore import UNSIGNED
from botocore.config import Config
import webdataset as wds
from tqdm import tqdm
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import io
import tifffile
import torch
from torchvision.transforms.functional import resize


async def download_sample_async(session, row, tar):
    """
    Downloads 5 fluorescent channels from S3, applies illumination correction
    (via division), resizes, and then writes them to 'tar' as a single NPY file.
    Raises an error if the illumination function is missing for any channel.
    """

    channels_corrected = []
    async with session.client("s3", config=Config(signature_version=UNSIGNED)) as s3:

        for channel in ["RNA", "ER", "AGP", "Mito", "DNA"]:
            s3_path_img = row.get(f"Image_{channel}", None)
            if not s3_path_img or pd.isna(s3_path_img):
                raise ValueError(
                    f"Raw image missing for channel '{channel}' at row {row.name}."
                )

            s3_path_illum = row.get(f"Illum_{channel}", None)
            if not s3_path_illum or pd.isna(s3_path_illum):
                raise ValueError(
                    f"No illumination function found for channel '{channel}' at row {row.name}.\n"
                    f"Raw image path = {s3_path_img}"
                )

            response_img = await s3.get_object(
                Bucket="cellpainting-gallery", Key=s3_path_img
            )
            img_data = await response_img["Body"].read()
            raw_img = tifffile.imread(io.BytesIO(img_data))

            response_illum = await s3.get_object(
                Bucket="cellpainting-gallery", Key=s3_path_illum
            )
            illum_data = await response_illum["Body"].read()
            illum = np.load(io.BytesIO(illum_data))

            # Apply correction: raw_img / illum
            epsilon = 1e-8  # Avoid divide-by-zero
            corrected_img = raw_img / (illum + epsilon)

            channels_corrected.append(corrected_img)

    if len(channels_corrected) == 0:
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
    """Process one tar shard"""
    with wds.TarWriter(str(tar_path)) as tar:
        tasks = [
            download_sample_async(session, row, tar) for _, row in shard_df.iterrows()
        ]
        await asyncio.gather(*tasks)


async def download_and_create_tar_async(
    split_df, output_dir, split_name, samples_per_tar=100
):
    """Downloads images and creates tar files for a split"""
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


async def main():
    subset = "0.01percent"
    df = pd.read_parquet(f"scripts/jump/data/subsets/subset_{subset}.parquet")

    train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42)

    output_dir = f"data/jump_{subset}"
    print(f"Overfit mode OFF: normal splits -> {output_dir}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Process each split
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for split_name, split_df in splits.items():
        await download_and_create_tar_async(
            split_df, output_dir, split_name, samples_per_tar=10
        )


if __name__ == "__main__":
    asyncio.run(main())
