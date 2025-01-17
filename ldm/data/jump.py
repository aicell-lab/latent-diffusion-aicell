import os
import webdataset as wds
import torch
from torchvision import transforms
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
        x_reshaped = x.reshape(x.shape[0], -1)

        min_vals = x_reshaped.min(dim=1, keepdim=True)[0]  # shape (C,1)
        max_vals = x_reshaped.max(dim=1, keepdim=True)[0]  # shape (C,1)

        # Do per-channel min-max
        x_reshaped = (x_reshaped - min_vals) / (max_vals - min_vals + 1e-8)
        return x_reshaped.reshape(x.shape)


class JumpWebDataset:
    def __init__(self, root="data/jump_wds", train=True, download=False, batch_size=1):
        """
        WebDataset wrapper for your dataset.
        Args:
          root: Path to the WebDataset shards
          train: If True, load from 'train' subdir, else from 'val' subdir
          download: (unused, for API compatibility)
          batch_size: how many samples per batch
        """
        split = "train" if train else "val"
        split_dir = os.path.join(root, split)

        # 1) Gather shard files
        shard_files = [
            f
            for f in os.listdir(split_dir)
            if f.startswith("shard_") and f.endswith(".tar")
        ]
        if not shard_files:
            raise RuntimeError(f"No shards found in {split_dir} matching shard_*.tar")

        # 2) Parse numeric shard indices (e.g., shard_000013.tar -> 13)
        shard_nums = []
        for sf in shard_files:
            # shard_000013.tar
            base = sf.split(".")[0]  # "shard_000013"
            idx_str = base.split("_")[1]  # "000013"
            idx_val = int(idx_str)  # 13
            shard_nums.append(idx_val)

        max_shard = max(shard_nums)
        min_shard = min(shard_nums)

        # For cpg0012, shards typically increment by 1. If you actually need step=10, adjust below.
        step = 1

        # 3) Build the brace expansion pattern
        # E.g. "shard_{000000..000036..1}.tar" if max_shard=36
        pattern = os.path.join(
            split_dir, f"shard_{{{min_shard:06d}..{max_shard:06d}..{step}}}.tar"
        )

        print(
            f"Found {len(shard_files)} shards in {split_dir}. "
            f"Using brace expansion: {pattern}"
        )

        self.transform = transforms.Compose(
            [
                NumpyLoader(),
                MinMaxNormalize(),
            ]
        )

        shuffle_buffer = 1000 if train else 0

        # 4) Build the WebDataset pipeline
        #    By passing the brace expansion pattern as a string, WebDataset tries to interpret
        #    it with its internal expansion logic (similar to shell expansion).
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

        # 5) If each shard is known to have a fixed # of samples, e.g. 10:
        samples_per_shard = 10
        self.length = len(shard_files) * samples_per_shard

    def _transform_image(self, img_bytes):
        """Transform the raw .npy bytes -> (C,H,W) tensor, then min-max normalize."""
        img = self.transform(img_bytes)
        return img

    def _format_batch_as_dict(self, batch):
        """Convert tuple -> {"image": <tensor>} for training loops."""
        images = batch[0]
        return {"image": images, "class_label": torch.zeros(len(images))}

    def __len__(self):
        return self.length
