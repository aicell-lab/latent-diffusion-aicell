import os
import json
import torch
import torchvision
from torchvision import transforms
import webdataset as wds
import numpy as np
import io


class NumpyLoader(torch.nn.Module):
    """Load numpy array from bytes"""

    def forward(self, x):
        img = np.load(io.BytesIO(x), allow_pickle=False)
        return torch.from_numpy(img).float()


class MinMaxNormalize(torch.nn.Module):
    def forward(self, x):
        # x is (C, H, W)
        # Flatten spatial dims => (C, H*W)
        x_reshaped = x.reshape(x.shape[0], -1)

        min_vals = x_reshaped.min(dim=1, keepdim=True)[0]  # shape (C,1)
        max_vals = x_reshaped.max(dim=1, keepdim=True)[0]  # shape (C,1)

        # Do per-channel min-max
        x_reshaped = (x_reshaped - min_vals) / (max_vals - min_vals + 1e-8)

        return x_reshaped.reshape(x.shape)


class JumpWebDataset:
    def __init__(self, root="data/jump_wds", train=True, download=False, batch_size=1):
        """
        WebDataset wrapper for JUMP Cell Painting dataset.
        Args:
            root: Path to the WebDataset shards
            train: If True, load training set, else load test set
            download: Unused, kept for compatibility
            batch_size: Batch size for batching samples
        """
        split = "train" if train else "val"
        split_dir = os.path.join(root, split)

        # Count number of shards from directory
        n_shards = len(list(os.listdir(split_dir))) - 1

        # Before figuring out the pattern, let's get the actual shard numbers
        shard_files = sorted(os.listdir(split_dir))
        shard_nums = [int(f.split("_")[1].split(".")[0]) for f in shard_files]
        max_shard = max(shard_nums)

        # Then create pattern with correct step size
        pattern = os.path.join(
            split_dir, f"shard_{{00000000..{max_shard:08d}..10}}.tar"
        )

        self.transform = transforms.Compose(
            [
                NumpyLoader(),
                MinMaxNormalize(),  # First normalize to [0,1]
            ]
        )

        # Create dataset pipeline
        shuffle_buffer = 1000 if train else 0

        self.dataset = (
            wds.WebDataset(
                pattern, nodesplitter=wds.shardlists.split_by_node, shardshuffle=False
            )
            .shuffle(shuffle_buffer)
            .to_tuple("image.npy")
            .map_tuple(self._transform_image)
            .batched(batch_size)
            .map(self._format_batch_as_dict)
        )

        # Calculate length from known shard structure
        samples_per_shard = 10
        self.length = (n_shards + 1) * samples_per_shard

    def _transform_image(self, img_bytes):
        """Transform numpy bytes to tensor"""
        img = self.transform(img_bytes)
        # img = img.permute(1, 2, 0)  # Convert to [H,W,C] format
        return img

    def _format_batch_as_dict(self, batch):
        """Convert batched tuple to dictionary format expected by model"""
        images = batch[0]  # Only one element in tuple
        return {"image": images, "class_label": torch.zeros(len(images))}

    def __len__(self):
        return self.length
