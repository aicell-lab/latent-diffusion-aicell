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
        self.dataset.data = self.dataset.data[:500]
        self.dataset.targets = self.dataset.targets[:500]

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


class MNISTWebDataset(torch.utils.data.Dataset):
    def __init__(self, root="data/mnist_wds", train=True, download=False):
        """
        WebDataset wrapper for MNIST that maintains compatibility with MNISTWrapper.
        Args:
            root: Path to the WebDataset shards
            train: If True, load training set, else load test set
            download: Unused, kept for compatibility
        """
        super().__init__()

        # Read metadata for length
        metadata_path = os.path.join(root, "metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Set up the dataset
        split = "train" if train else "test"
        n_shards = self.metadata[f"{split}_shards"] - 1  # -1 because range is inclusive

        # Use WebDataset's native braced pattern
        pattern = os.path.join(root, f"mnist-{split}-{{00000..{n_shards:05d}}}.tar")

        # Create dataset pipeline - note we're not using .map() here
        self.dataset = (
            wds.WebDataset(pattern)
            .decode("pil")
            .to_tuple("png", "cls")
            .batched(1)  # This helps with deterministic behavior
        )

        # Store iterator for __getitem__
        self.iterator = None

        # Store length based on metadata
        self.length = (
            self.metadata["total_train_samples"]
            if train
            else self.metadata["total_test_samples"]
        )

        # Define transform
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[-1.0], std=[2.0]
                ),  # Scale to [-1,1]
            ]
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.iterator is None:
            self.iterator = iter(self.dataset)

        try:
            # Get next batch (which is actually a single sample due to batched(1))
            image, label = next(self.iterator)

            # Remove the batch dimension added by batched(1)
            image = image[0]
            label = label[0]

            # Convert to grayscale if it's RGB
            if image.mode != "L":
                image = image.convert("L")

            # Transform image
            img = self.transform(image)  # [1,H,W]
            img = img.permute(1, 2, 0)  # Convert to [H,W,1]

            return {"image": img, "class_label": label}

        except StopIteration:
            self.iterator = iter(self.dataset)
            image, label = next(self.iterator)

            # Remove the batch dimension
            image = image[0]
            label = label[0]

            # Convert to grayscale if it's RGB
            if image.mode != "L":
                image = image.convert("L")

            # Transform image
            img = self.transform(image)  # [1,H,W]
            img = img.permute(1, 2, 0)  # Convert to [H,W,1]

            return {"image": img, "class_label": label}
