import torch
import torchvision


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
