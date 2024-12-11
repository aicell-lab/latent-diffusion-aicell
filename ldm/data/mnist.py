import os
import json

import torch
import torchvision
import webdataset as wds
import numpy as np


class MNISTWrapper(torch.utils.data.Dataset):
    def __init__(self, root="data/mnist", train=True, download=True):
        # Create MNIST dataset directly instead of receiving it
        self.dataset = torchvision.datasets.MNIST(
            root=root, train=train, download=download
        )
        # Take only first 5000 samples for now
        # self.dataset.data = self.dataset.data[:500]
        # self.dataset.targets = self.dataset.targets[:500]

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[-1.0], std=[2.0]
                ),  # Scale from [0,1] to [-1,1]
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)  # This gives us [C,H,W]
        img = img.permute(1, 2, 0)  # Convert to [H,W,C] to match model's expectation
        return {"image": img, "class_label": label}


class MNISTWebDataset:
    def __init__(
        self, root="data/mnist_wds", train=True, download=False, batch_size=12
    ):
        """
        WebDataset wrapper for MNIST that provides efficient streaming access.
        Args:
            root: Path to the WebDataset shards
            train: If True, load training set, else load test set
            download: Unused, kept for compatibility
            batch_size: Batch size for batching samples
        """
        # Read metadata
        metadata_path = os.path.join(root, "metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Set up dataset parameters
        split = "train" if train else "test"
        n_shards = self.metadata[f"{split}_shards"] - 1
        pattern = os.path.join(root, f"mnist-{split}-{{00000..{n_shards:05d}}}.tar")

        # Define transforms
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[-1.0], std=[2.0]),
            ]
        )

        # Create dataset pipeline
        shuffle_buffer = 1000 if train else 0

        self.dataset = (
            wds.WebDataset(pattern, nodesplitter=wds.shardlists.split_by_node)
            .shuffle(shuffle_buffer)
            .decode("pil")
            .to_tuple("png", "cls")
            .map_tuple(self._transform_image, None)
            .batched(batch_size)
            .map(self._format_batch_as_dict)
        )

        # Store length for compatibility
        self.length = self.metadata[f"total_{split}_samples"]
        # Uncomment for testing with subset
        # self.length = min(500, self.length)

    def _transform_image(self, img):
        """Transform PIL image to tensor with correct format"""
        if img.mode != "L":
            img = img.convert("L")
        img = self.transform(img)  # [1,H,W]
        img = img.permute(1, 2, 0)  # Convert from [C,H,W] to [H,W,C]
        return img

    def _format_batch_as_dict(self, batch):
        """Convert batched tuple to dictionary format expected by model"""
        images, labels = batch
        return {"image": images, "class_label": labels}

    def __len__(self):
        return self.length
