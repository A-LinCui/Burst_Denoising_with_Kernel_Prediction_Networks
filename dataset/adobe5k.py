import os
from typing import Optional, Union

import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision import transforms
from torch.utils.data import Dataset, Subset
from extorch.vision.dataset import CVDataset


class Adobe5K(CVDataset):
    def __init__(self, data_dir: str, train_ratio: float, random_split: bool, 
            train_transform: transforms.Compose = None,
            test_transform: transforms.Compose = None):
        super(Adobe5K, self).__init__(data_dir, train_transform, test_transform)
        assert 0 < train_ratio < 1, "train_ratio should be in (0., 1.)"
        train_dataset = Adobe5KDataset(data_dir, self.transforms["train"])

        total_num = len(train_dataset)
        indices = (torch.randperm if random_split else np.arange)(total_num).tolist()

        train_dataset = Subset(train_dataset, indices[:int(train_ratio * total_num)])
        train_dataset.transforms = self.transforms["train"]
        self.datasets["train"] = train_dataset

        test_dataset = Adobe5KDataset(data_dir, test_transform)
        test_dataset = Subset(test_dataset, indices[int(train_ratio * total_num):])
        test_dataset.transforms = self.transforms["test"]
        self.datasets["test"] = test_dataset


class Adobe5KDataset(Dataset):
    def __init__(self, root: str, transform: Optional[transforms.Compose] = None, 
            target_transform: Optional[transforms.Compose] = None) -> None:
        super(Adobe5KDataset, self).__init__()
        self.root = root
        self.data_list = os.listdir(self.root)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Union[np.ndarray, Tensor]:
        image = cv2.imread(os.path.join(self.root, self.data_list[index]), cv2.IMREAD_GRAYSCALE)
        if self.transform:
            return self.transform(image)
        return image

    def __len__(self):
        return len(self.data_list)
