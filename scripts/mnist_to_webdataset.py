import os
from torchvision.datasets import MNIST
import webdataset as wds
import io
import json
from tqdm import tqdm


def convert_mnist_to_webdataset(output_dir="data/mnist_wds", samples_per_shard=1000):
    """Convert MNIST dataset to WebDataset format."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Download and load MNIST
    train_dataset = MNIST("data/mnist", train=True, download=True)
    test_dataset = MNIST("data/mnist", train=False, download=True)

    def write_dataset(dataset, name):
        n_shards = len(dataset) // samples_per_shard + (
            1 if len(dataset) % samples_per_shard != 0 else 0
        )

        for shard_idx in tqdm(range(n_shards), desc=f"Converting {name}"):
            # Create tar writer for this shard
            shard_name = f"{name}-{shard_idx:05d}.tar"
            sink = wds.TarWriter(os.path.join(output_dir, shard_name))

            # Calculate start and end indices for this shard
            start_idx = shard_idx * samples_per_shard
            end_idx = min(start_idx + samples_per_shard, len(dataset))

            # Write samples
            for idx in range(start_idx, end_idx):
                img, label = dataset[idx]

                # Convert image to bytes
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes = img_bytes.getvalue()

                # Create sample dict
                sample = {
                    "__key__": f"{idx:08d}",
                    "png": img_bytes,
                    "cls": label,
                    "json": json.dumps({"label": int(label)}),
                }

                # Write to tar file
                sink.write(sample)

            sink.close()

    # Convert both train and test sets
    write_dataset(train_dataset, "mnist-train")
    write_dataset(test_dataset, "mnist-test")

    # Write metadata
    metadata = {
        "train_shards": len(train_dataset) // samples_per_shard
        + (1 if len(train_dataset) % samples_per_shard != 0 else 0),
        "test_shards": len(test_dataset) // samples_per_shard
        + (1 if len(test_dataset) % samples_per_shard != 0 else 0),
        "samples_per_shard": samples_per_shard,
        "total_train_samples": len(train_dataset),
        "total_test_samples": len(test_dataset),
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    convert_mnist_to_webdataset()
