import torch
import torchvision
from abc import abstractmethod
from torch.utils.data import Dataset, ConcatDataset, ChainDataset, IterableDataset


class Txt2ImgIterableBaseDataset(IterableDataset):
    """
    Define an interface to make the IterableDatasets for text2img data chainable
    """

    def __init__(self, num_records=0, valid_ids=None, size=256):
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size

        print(f"{self.__class__.__name__} dataset contains {self.__len__()} examples.")

    def __len__(self):
        return self.num_records

    @abstractmethod
    def __iter__(self):
        pass


class MNISTWrapper(torch.utils.data.Dataset):
    def __init__(self, root="data/mnist", train=True, download=True):
        # Create MNIST dataset directly instead of receiving it
        self.dataset = torchvision.datasets.MNIST(
            root=root, train=train, download=download
        )
        # Take only first 100 samples for testing
        self.dataset.data = self.dataset.data[:100]
        self.dataset.targets = self.dataset.targets[:100]

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
