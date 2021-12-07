import skimage
import torch
from torch import Tensor
import extorch.utils as utils


def sRGBTransfer(input: Tensor) -> Tensor:
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
    output = 255. * output.unsqueeze(-1).detach().cpu().numpy()
    target = 255. * target.unsqueeze(-1).detach().cpu().numpy()

    psnrs = utils.AverageMeter()
    for (out, label) in zip(output, target):
        psnr = skimage.metrics.peak_signal_noise_ratio(out, label, data_range = 255.)
        psnrs.update(psnr)
    return psnrs.avg


def cal_ssim(output: Tensor, target: Tensor) -> float:
    output = 255. * output.unsqueeze(-1).detach().cpu().numpy()
    target = 255. * target.unsqueeze(-1).detach().cpu().numpy()

    ssims = utils.AverageMeter()
    for (out, label) in zip(output, target):
        ssim = skimage.metrics.structural_similarity(out, label, data_range = 255., multichannel = True)
        ssims.update(ssim)
    return ssims.avg
