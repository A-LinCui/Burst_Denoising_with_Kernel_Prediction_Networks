import skimage
import torch
from torch import Tensor
import extorch.utils as utils


def sRGBTransfer(input: Tensor) -> Tensor:
    r"""
    The applied sRGD transfer function.

    Args:
        input (Tensor): The input images.

    Returns:
        res (Tensor): The images after transfermation.
    """
    threshold = 0.0031308
    a = 0.055
    mult = 12.92
    gamma = 2.4
    res = torch.zeros_like(input)
    mask = input > threshold
    res[mask] = (1 + a) * torch.pow(input[mask] + 0.001, 1.0 / gamma) - a
    res[~mask] = input[~mask] * mult
    return res


def cal_psnr(output: Tensor, target: Tensor) -> float:
    r"""
    Calculate PSNR between images.

    Args:
        output (Tensor): The batch of images with pixel values in [0, 1]. 
                         Shape: [b, h, w].
        target (Tensor): The batch of reference images with pixel values in [0, 1]. 
                         Shape: [b, h, w].

    Returns:
        psnr (float): Average PSNR value.
    """
    output = 255. * output.unsqueeze(-1).detach().cpu().numpy()
    target = 255. * target.unsqueeze(-1).detach().cpu().numpy()

    psnrs = utils.AverageMeter()
    for (out, label) in zip(output, target):
        psnr = skimage.metrics.peak_signal_noise_ratio(out, label, data_range = 255.)
        psnrs.update(psnr)
    return psnrs.avg


def cal_ssim(output: Tensor, target: Tensor) -> float:
    r"""
    Calculate SSIM between images.

    Args:
        output (Tensor): The batch of images with pixel values in [0, 1]. 
                         Shape: [b, h, w].
        target (Tensor): The batch of reference images with pixel values in [0, 1]. 
                         Shape: [b, h, w].

    Returns:
        ssim (float): Average SSIM value.
    """
    output = 255. * output.unsqueeze(-1).detach().cpu().numpy()
    target = 255. * target.unsqueeze(-1).detach().cpu().numpy()

    ssims = utils.AverageMeter()
    for (out, label) in zip(output, target):
        ssim = skimage.metrics.structural_similarity(out, label, data_range = 255., multichannel = True)
        ssims.update(ssim)
    return ssims.avg
