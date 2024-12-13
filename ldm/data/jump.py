import os
import json
import torch
import torchvision
import webdataset as wds
import numpy as np
import io


class JumpWebDataset:
    def __init__(
        self, root="data/jump_wds", train=True, download=False, batch_size=1
    ):  # Default batch_size=1 due to large images
        """
        WebDataset wrapper for JUMP Cell Painting dataset.
        Args:
            root: Path to the WebDataset shards
            train: If True, load training set, else load test set
            download: Unused, kept for compatibility
            batch_size: Batch size for batching samples
        """
        # Jump doesn't have metadata.json, need to calculate from directory structure
        split = (
            "train" if train else "val"
        )  # Note: Jump uses 'val' where MNIST uses 'test'
        split_dir = os.path.join(root, split)

        # Count number of shards from directory
        n_shards = len(list(os.listdir(split_dir))) - 1  # -1 for zero-based indexing

        # Pattern matches jump shard naming: shard_00000000.tar vs mnist-train-00000.tar
        pattern = os.path.join(split_dir, f"shard_{{00000000..{n_shards:08d}}}.tar")

        # No transforms needed - normalization was done during download
        # MNIST needed ToTensor() and Normalize(), but Jump data is already normalized numpy arrays

        # Create dataset pipeline
        shuffle_buffer = 1000 if train else 0

        self.dataset = (
            wds.WebDataset(
                pattern, nodesplitter=wds.shardlists.split_by_node, shardshuffle=False
            )
            .shuffle(shuffle_buffer)
            .decode()  # No "pil" arg needed - we're loading .npy files not images
            .to_tuple(
                "image.npy"
            )  # Only one tuple element - no labels needed for autoencoder
            .map_tuple(self._transform_image)  # No second transform needed for labels
            .batched(batch_size)
            .map(self._format_batch_as_dict)
        )

        # Calculate length from known shard structure
        # Each shard has 10 samples (from download script)
        samples_per_shard = 10
        self.length = (n_shards + 1) * samples_per_shard

    def _transform_image(self, img_bytes):
        """Transform numpy bytes to tensor
        Different from MNIST:
        - Input is .npy bytes instead of PIL image
        - No need to convert to grayscale
        - No normalization needed (done during download)
        - Different channel order handling (5 channels vs 1)
        """
        # Load numpy array from bytes
        img = np.load(io.BytesIO(img_bytes))

        # Convert to tensor - already in [C,H,W] format from download script
        img = torch.from_numpy(img)

        # Convert to [H,W,C] for model consistency with MNIST
        img = img.permute(1, 2, 0)

        return img

    def _format_batch_as_dict(self, batch):
        """Convert batched tuple to dictionary format expected by model
        Different from MNIST:
        - Only handles images (no labels)
        - Uses dummy label 0 since autoencoder doesn't need labels
        """
        images = batch[0]  # Only one element in tuple
        return {"image": images, "class_label": torch.zeros(len(images))}

    def __len__(self):
        return self.length
