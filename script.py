import torch
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib import cm


def draw(target: Tensor, output: Tensor, burst: Tensor, path: str) -> None:#
    r"""
    Draw the denoising result and save the figure under given path.

    Args:
        target (Tensor): The reference groundtruth. Shape: [h, w].
        output (Tensor): The output image. Shape: [h, w].
        burst (Tensor): The burst of input images. Shape: [N, h, w].
        path (str): Path to save the figure.
    """
    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(target.unsqueeze(-1).detach().cpu().numpy(), cmap = cm.gray)
    plt.title("target")
    plt.axis("off")
    plt.subplot(1, 4, 2)
    plt.imshow(all_burst[0].unsqueeze(-1).detach().cpu().numpy(), cmap = cm.gray)
    plt.title("input")
    plt.axis("off")
    plt.subplot(1, 4, 3)
    plt.imshow(output.unsqueeze(-1).detach().cpu().numpy(), cmap = cm.gray)
    plt.title("output")
    plt.axis("off")
    plt.subplot(1, 4, 4)
    plt.imshow(torch.mean(burst, 0).unsqueeze(-1).detach().cpu().numpy(), cmap = cm.gray)
    plt.title("average")
    plt.axis("off")
    plt.savefig(path, dpi = 600)
