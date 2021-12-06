from typing import List

import numpy as np
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from extorch.nn.module.operation import ConvReLU


class KPNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> None:
        super(KPNBlock, self).__init__()
        self.op = nn.Sequential(
                ConvReLU(in_channels, out_channels, kernel_size, stride, padding),
                ConvReLU(out_channels, out_channels, kernel_size, stride, padding),
                ConvReLU(out_channels, out_channels, kernel_size, stride, padding)
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.op(input)


class KPN(nn.Module):
    def __init__(self, burst_length: int, kernel_size: List[int], core_bias: bool = False) -> None:
        super(KPN, self).__init__()
        self.burst_length = burst_length
        self.blind_est = blind_est
        self.kernel_size = kernel_size
        self.core_bias = core_bias
        
        in_channels = self.burst_length + 1
        out_channels = np.sum(np.array(self.kernel_size) ** 2)) * self.burst_length
        if self.core_bias:
            out_channels += self.burst_length

        down_sample_out_channels = [64, 128, 256, 512, 512]
        self.downsample_layers = nn.ModuleList(
                KPNBlock(in_channels = in_channels if i == 0 else down_sample_out_channels[i - 1],
                         out_channels = down_sample_out_channels[i],
                         kernel_size = 3, 
                         stride = 1, 
                         padding = 1
                ) for i in range(len(down_sample_out_channels))
        )

        self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)

        up_sample_out_channels = [512, 256, out_channel]
        self.upsample_layers = nn.ModuleList(
            KPNBlock(in_channels = down_sample_out_channels[-i - 1] + down_sample_out_channels[-i - 2],
                     out_channels = up_sample_out_channels[i],
                     kernel_size = 3, 
                     stride = 1, 
                     padding = 1
            ) for i in range(len(up_sample_out_channels))
        )

        self.conv = nn.Conv2d(out_channel, out_channel, 1, 1, 0)
        self.predict_kernel = PredictionKernelConv(kernel_size, self.core_bias)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.)

    def forward(self, input: Tensor) -> Tensor:
        down_sample_outputs = [input]
        for down_block in self.downsample_layers:
            down_sample_outputs.append(self.avgpool(down_block(down_sample_outputs[-1])))

        outputs = down_sample_outputs[-1]
        for i in range(len(self.upsample_blocks)):
            outputs = self.upsample_blocks[i](
                    torch.cat([
                        down_sample_outputs[- i - 2], 
                        F.interpolate(outputs, scale_factor = 2, mode = "bilinear")
                        ], dim = 1)
            )

        outputs = self.conv(F.interpolate(outputs, scale_factor = 2, mode = "bilinear"))
        return self.predict_kernel(outputs)


class PredictionKernelConv(nn.Module):
    def __init__(self, kernel_size: List[int], bias: bool = False):
        super(PredictionKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.bias = bias

    def forward(self, input: Tensor):
        raise NotImplementedError
