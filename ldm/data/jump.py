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
    """Normalize tensor to [0,1] range"""

    def forward(self, x):
        # Handle each channel separately to preserve relative intensities
        x_norm = torch.zeros_like(x)
        for c in range(x.shape[-1]):  # Assuming [H,W,C] format
            channel = x[..., c]
            min_val = channel.min()
            max_val = channel.max()
            if max_val - min_val > 0:  # Avoid division by zero
                x_norm[..., c] = (channel - min_val) / (max_val - min_val)
            else:
                x_norm[..., c] = (
                    channel - min_val
                )  # If constant channel, normalize to 0
        return x_norm


class EfficientMinMaxNormalize(torch.nn.Module):
    def forward(self, x):
        # Reshape to [C, H*W] to process all channels at once
        x_reshaped = x.reshape(x.shape[-1], -1)

        # Calculate min/max for all channels simultaneously
        min_vals = x_reshaped.min(dim=1, keepdim=True)[0]
        max_vals = x_reshaped.max(dim=1, keepdim=True)[0]

        # Normalize in-place
        x_reshaped = (x_reshaped - min_vals) / (max_vals - min_vals + 1e-8)

        # Reshape back
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

        # Define transforms: MinMax normalize to [0,1] first, then normalize to [-1,1]
        self.transform = transforms.Compose(
            [
                NumpyLoader(),
                EfficientMinMaxNormalize(),  # First normalize to [0,1]
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5, 0.5, 0.5],  # This shifts [0,1] to [-1,1]
                    std=[0.5, 0.5, 0.5, 0.5, 0.5],  # This scales it to [-1,1]
                ),
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
        img = img.permute(1, 2, 0)  # Convert to [H,W,C] format
        return img

    def _format_batch_as_dict(self, batch):
        """Convert batched tuple to dictionary format expected by model"""
        images = batch[0]  # Only one element in tuple
        return {"image": images, "class_label": torch.zeros(len(images))}

    def __len__(self):
        return self.length
