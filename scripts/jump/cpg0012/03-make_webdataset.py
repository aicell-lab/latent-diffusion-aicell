#!/usr/bin/env python3

import os
import csv
import numpy as np
import tifffile
import webdataset as wds
from tqdm import tqdm
import argparse
import random


def load_5ch_stack(row):
    """
    Given a row with columns channelDNA, channelER, channelRNA, channelAGP, channelMito,
    load each TIF, stack into (5,H,W) NumPy array.
    """
    channel_paths = [
        row["channelDNA"],
        row["channelER"],
        row["channelRNA"],
        row["channelAGP"],
        row["channelMito"],
    ]
    stack = []
    for path in channel_paths:
        img = tifffile.imread(path).astype(np.float32)  # read TIF as float32
        stack.append(img)

    stack_5ch = np.stack(stack, axis=0)  # shape: (5, H, W)
    return stack_5ch


def create_writer(out_dir, shard_id):
    """
    Create a TarWriter for a given shard ID in a specified output directory.
    """
    fname = os.path.join(out_dir, f"shard_{shard_id:06d}.tar")
    return wds.TarWriter(fname, compress=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs_csv",
        type=str,
        default="pairs_allplates.csv",
        help="CSV file with columns [channelDNA,channelER,channelRNA,channelAGP,channelMito,BROAD_ID,CPD_SMILES,etc.]",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="jump_wds",
        help="Output directory to store the train/val/test subfolders",
    )
    parser.add_argument(
        "--samples_per_shard",
        type=int,
        default=100,
        help="How many samples per shard (per split)",
    )
    parser.add_argument(
        "--train_frac", type=float, default=0.8, help="Fraction of data for training"
    )
    parser.add_argument(
        "--val_frac", type=float, default=0.1, help="Fraction of data for validation"
    )
    parser.add_argument(
        "--test_frac", type=float, default=0.1, help="Fraction of data for test"
    )
    parser.add_argument(
        "--max_shards",
        type=int,
        default=None,
        help="Optional cap on the number of shards per split (for testing)",
    )
    args = parser.parse_args()

    # 1) Read rows from pairs CSV
    with open(args.pairs_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_samples = len(rows)
    print(f"Found {total_samples} rows in {args.pairs_csv}.")

    # 2) Shuffle the rows so the splits are random
    random.shuffle(rows)

    # 3) Compute split sizes
    train_size = int(args.train_frac * total_samples)
    val_size = int(args.val_frac * total_samples)
    # Remainder goes to test
    test_size = total_samples - train_size - val_size

    train_rows = rows[:train_size]
    val_rows = rows[train_size : train_size + val_size]
    test_rows = rows[train_size + val_size :]

    print(
        f"Split => Train: {len(train_rows)}, Val: {len(val_rows)}, Test: {len(test_rows)}"
    )

    # 4) Create subfolders: train/, val/, test/
    train_dir = os.path.join(args.out_dir, "train")
    val_dir = os.path.join(args.out_dir, "val")
    test_dir = os.path.join(args.out_dir, "test")

    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    # We'll define a helper function to write rows into shards
    def write_split(rows_split, split_out_dir, max_shards=None):
        """
        rows_split: subset of rows (train, val, or test)
        split_out_dir: e.g. jump_wds/train
        max_shards: optional limit on number of shards
        """
        shard_id = 0
        sample_in_shard = 0

        writer = create_writer(split_out_dir, shard_id)

        for i, row in enumerate(
            tqdm(rows_split, desc=f"Creating WDS in {split_out_dir}")
        ):
            try:
                stack_5ch = load_5ch_stack(row)
            except Exception as e:
                print(f"Warning: Error loading row {i}, skipping. {e}")
                continue

            sample_key = f"{shard_id}-{sample_in_shard}"
            sample_dict = {
                "__key__": sample_key,
                "image.npy": stack_5ch,
            }

            # Optionally store SMILES or compound ID
            meta = {
                "BROAD_ID": row.get("BROAD_ID", ""),
                "CPD_SMILES": row.get("CPD_SMILES", ""),
            }
            sample_dict["meta.json"] = meta

            writer.write(sample_dict)
            sample_in_shard += 1

            if sample_in_shard >= args.samples_per_shard:
                writer.close()
                shard_id += 1
                if max_shards is not None and shard_id >= max_shards:
                    print("Reached max_shards limit, stopping.")
                    break
                writer = create_writer(split_out_dir, shard_id)
                sample_in_shard = 0

        writer.close()
        print(f"Finished writing {split_out_dir} with shard_id={shard_id}.")

    # 5) Write each split
    write_split(train_rows, train_dir, max_shards=args.max_shards)
    write_split(val_rows, val_dir, max_shards=args.max_shards)
    write_split(test_rows, test_dir, max_shards=args.max_shards)

    print("All done!")


if __name__ == "__main__":
    main()
