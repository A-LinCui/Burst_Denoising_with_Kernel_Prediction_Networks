import matplotlib.pyplot as plt
from matplotlib import cm


def draw(target, input, output, path):
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(target.unsqueeze(-1).detach().cpu().numpy(), cmap = cm.gray)
    plt.title("target")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(input.unsqueeze(-1).detach().cpu().numpy(), cmap = cm.gray)
    plt.title("input")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(output.unsqueeze(-1).detach().cpu().numpy(), cmap = cm.gray)
    plt.title("output")
    plt.axis("off")
    plt.savefig(path, dpi = 600)
