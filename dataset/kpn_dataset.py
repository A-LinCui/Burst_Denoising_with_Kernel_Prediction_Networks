import os
from typing import Dict, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.nn.functional import adaptive_avg_pool2d
from extorch.vision.dataset import CVDataset
from extorch.vision.transforms import AdaptiveRandomCrop, AdaptiveCenterCrop

from adobe5k import Adobe5K

                   
class KPNTransform(nn.Module):
    r"""
    KPN transform for synthetic training data.

    Args:
        burst_num (int): Number of images in the input burst, denoted by ``N`` in the paper.
        downsample (int): Downsample patches in each dimension using a box filter, 
                          which reduces noise and compression artifacts.
        blind (bool): Whether to estimate blindly.
        misalignment (int): Maximum misalignment.
        max_translational_shift (int): Maximum number of pixels for translational shift 
                                       after downsampling relative to the reference.
        train (bool): Only apply the horizontal flpping in training mode.
    """
    def __init__(self, burst_num: int, downsample: int, blind: bool, 
            misalignment: int, max_translational_shift: int, train: bool) -> None:
        super(KPNTransform, self).__init__()
        self.burst_num = burst_num
        self.downsample = downsample

        # Only apply the horizontal flpping in training mode
        self.train = train
        if self.train:
            self.horizontal_flip = transforms.RandomHorizontalFlip(p = 0.5)

        self.blind = blind

        self.adaptive_crop = AdaptiveRandomCrop(downsample * misalignment)
        self.misalignment_crop = AdaptiveRandomCrop(downsample * max_translational_shift)

        self.center_crop = AdaptiveCenterCrop(downsample * max_translational_shift)
        self.center_crop_2 = AdaptiveCenterCrop(
                downsample * (max_translational_shift - misalignment))
        
    def forward(self, img: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Args:
           img (Tensor): The origin image ([1, H, W]). At present, we only support gray images.

        Returns:
            burst_noise (Tensor): Input burst with noise.
            target (Tensor): The groundtruth image.
            white_level (Tensor): White level.
        """
        # Step 1: Generate the target image
        img_burst = [self.center_crop(img)]

        # Step 2: Generate a synthetic burst of N frames
        """
        It is critical to also simulate complete alignment failure to provide robustness in the 
        presence of occlusion or large scene motion. Some real bursts will be easy to align and
        some hard, so for each burst we pick an approximate number of misaligned frames 
        n ~ Poisson(λ). Then for each alternate frame in that burst, we sample an
        alternate frame in that burst, we sample a coin flip with probability n/N to decide 
        whether to apply a translational shift after downsampling relative to the reference.
        """
        for i in range(1, self.burst_num):
            # Simulate complete alignment failure
            if np.random.binomial(1, min(1., np.random.poisson(lam = 1.5) / self.burst_num)):
                img_burst.append(self.misalignment_crop(img))
            else:
                img_burst.append(self.center_crop_2(self.adaptive_crop(img)))

        img_burst = torch.stack(img_burst, dim = 0)
        
        # Step 3: Downsampling
        w, h = F._get_image_size(img)
        img_burst = adaptive_avg_pool2d(img_burst, (w // self.downsample, h // self.downsample))

        # Step 4: Hotizontal flipping
        if self.train:
            img_burst = self.horizontal_flip(img_burst)

        target = torch.clone(img_burst[0]) # The origin image is the first in the burst frames

        white_level = torch.pow(10, -torch.rand((1, 1, 1))).type_as(img_burst)
        img_burst = white_level * img_burst

        burst_noise = self.sim_burst_noise(img_burst, target)

        white_level = white_level.unsqueeze(0)
        return burst_noise, target, white_level

    def sim_burst_noise(self, img_burst: Tensor, target: Tensor) -> Tensor:
        r"""
        Add simulated noise on the input burst.

        Args:
            img_burst (Tensor): Input burst.
            target (Tensor): The groundtruth.

        Returns:
            burst_noise (Tensor): Input burst with simulated noise.
        """
        sigma_read = torch.from_numpy(
                np.power(10, np.random.uniform(-3.0, -1.5, (1, 1, 1)))
        ).type_as(img_burst)
        sigma_shot = torch.from_numpy(
                np.power(10, np.random.uniform(-4.0, -2.0, (1, 1, 1)))
        ).type_as(img_burst)

        sigma_read_com = sigma_read.expand_as(img_burst)
        sigma_shot_com = sigma_shot.expand_as(img_burst)

        burst_noise = torch.normal(img_burst, torch.sqrt(sigma_read_com ** 2 + 
            img_burst * sigma_shot_com)).type_as(img_burst)
        burst_noise = torch.clamp(burst_noise, 0., 1.)

        if not self.blind:
            sigma_read_est = sigma_read.view(1, 1).expand_as(target)
            sigma_shot_est = sigma_shot.view(1, 1).expand_as(target)
            sigma_estimate = torch.sqrt(
                sigma_read_est ** 2 + sigma_shot_est.mul(
                    torch.max(
                        torch.stack([burst_noise[0], torch.zeros_like(burst_noise[0])], dim = 0), 
                    dim = 0)[0])
            )
            burst_noise = torch.cat([burst_noise, sigma_estimate.unsqueeze(0)], dim = 0)
        
        return burst_noise


class KPNDataset(CVDataset):
    def __init__(self, data_dir: str, 
                 base_dataset_cls, base_dataset_cfg: Optional[Dict] = None,
                 kpn_transform_kwargs: Dict = {
                     "burst_num": 8,
                     "downsample": 4,
                     "blind": False,
                     "misalignment": 2,
                     "max_translational_shift": 16
                     },
                 train_transform: transforms.Compose = None,
                 test_transform: transforms.Compose = None) -> None:

        # transform setting
        self.kpn_transform_kwargs = kpn_transform_kwargs
        train_transform = train_transform or self.default_transform["train"]
        test_transform = test_transform or self.default_transform["test"]

        super(KPNDataset, self).__init__(data_dir, train_transform, test_transform)
        base_datasets = base_dataset_cls(data_dir, train_transform = self.transforms["train"], 
                test_transform = self.transforms["test"], **base_dataset_cfg)
        self.datasets["train"] = base_datasets.splits["train"]
        self.datasets["test"] = base_datasets.splits["test"]

    @property
    def default_transform(self) -> Dict[str, transforms.Compose]:
        train_transform = transforms.Compose([
                transforms.ToTensor(),
                KPNTransform(train = True, **self.kpn_transform_kwargs)]
        )

        test_transform = transforms.Compose([
                transforms.ToTensor(),
                KPNTransform(train = False, **self.kpn_transform_kwargs)]
        )

        default_transforms = {
                "train": train_transform,
                "test": test_transform
        }
        return default_transforms


if __name__ == "__main__":
    dataset = KPNDataset(data_dir = "/mnt/c/Users/HUAWEI/Desktop/Adobe5K_gray", 
                         kpn_transform_kwargs = {
                             "burst_num": 8,
                             "downsample": 1,
                             "blind": False,
                             "misalignment": 2,
                             "max_translational_shift": 10},
                         base_dataset_cls = Adobe5K,
                         base_dataset_cfg = {
                             "train_ratio": 0.5, 
                             "random_split": False}
                         )

    for i, data in enumerate(dataset.splits["test"]):

        if i < 20:
            continue

        from matplotlib import cm
        import matplotlib.pyplot as plt
        for i, _data in enumerate(data[0]):
            _data = _data.numpy().transpose(1, 2, 0)
            plt.figure()
            plt.imshow(_data, cmap = cm.gray)
            plt.savefig("/mnt/e/noise_{}.png".format(i), dpi = 600)
        plt.figure()
        plt.imshow(data[1].numpy().transpose(1, 2, 0), cmap = cm.gray)
        plt.savefig("/mnt/e/target.png", dpi = 600)

        break
