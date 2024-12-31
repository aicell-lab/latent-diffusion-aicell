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

OVERFIT = True  # Set to True if you want 1 sample for train, 1 for val, 1 for test


async def download_sample_async(session, row, tar):
    channels = []

    # no brightfield for now
    # the order is ["RNA", "ER", "AGP", "Mito", "DNA"]
    for channel in ["RNA", "ER", "AGP", "Mito", "DNA"]:
        s3_path = row.get(f"Image_{channel}", None)
        if not s3_path:
            continue

        async with session.client(
            "s3", config=Config(signature_version=UNSIGNED)
        ) as s3:
            response = await s3.get_object(Bucket="cellpainting-gallery", Key=s3_path)
            img_data = await response["Body"].read()
            img = tifffile.imread(io.BytesIO(img_data))
            channels.append(img)

    # Stack into 5-channel array
    combined = np.stack(channels, axis=0).astype(np.float32)  # shape = (5,H,W)

    # Resize the stacked array (all channels together)
    combined_tensor = torch.from_numpy(combined)
    resized_tensor = resize(combined_tensor, [512, 512])  # (5,512,512)

    # Save to buffer
    buffer = io.BytesIO()
    np.save(buffer, resized_tensor.numpy())
    buffer.seek(0)

    # Write to tar
    tar.write({"__key__": f"sample_{row.name:08d}", "image.npy": buffer.getvalue()})


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
    subset = "0.001percent"
    df = pd.read_parquet(f"scripts/jump/data/subsets/subset_{subset}.parquet")

    if OVERFIT:
        # Just pick the first 3 rows if available
        if len(df) < 3:
            raise ValueError("Not enough rows to pick 3 samples (train/val/test).")

        train_df = df.iloc[[0]].copy()
        val_df = df.iloc[[1]].copy()
        test_df = df.iloc[[2]].copy()

        output_dir = "data/jump_overfit"
        print(f"Overfit mode ON: 1 sample each -> {output_dir}")
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    else:
        train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.1, random_state=42
        )

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
