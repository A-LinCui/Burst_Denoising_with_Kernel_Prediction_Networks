from typing import List

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from extorch.nn import ConvReLU


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
    def __init__(self, burst_length: int, kernel_size: int) -> None:
        super(KPN, self).__init__()
        self.burst_length = burst_length
        
        in_channels = self.burst_length
        out_channels = kernel_size ** 2 * self.burst_length

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

        up_sample_out_channels = [512, 256, out_channels]
        self.upsample_layers = nn.ModuleList(
            KPNBlock(in_channels = down_sample_out_channels[-i - 1] + down_sample_out_channels[-i - 2],
                     out_channels = up_sample_out_channels[i],
                     kernel_size = 3, 
                     stride = 1, 
                     padding = 1
            ) for i in range(len(up_sample_out_channels))
        )

        self.conv = nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        self.predict_kernel = PredictionKernelConv()

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.)

    def forward(self, input: Tensor, white_level: Tensor) -> Tensor:
        down_sample_outputs = [input]
        for i, down_block in enumerate(self.downsample_layers):
            if i == 0:
                down_sample_outputs.append(down_block(down_sample_outputs[-1]))
            else:
                down_sample_outputs.append(self.avgpool(down_block(down_sample_outputs[-1])))

        outputs = down_sample_outputs[-1]

        for i, upsample_layer in enumerate(self.upsample_layers):
            interpolate_size = down_sample_outputs[- i - 2].shape[2:]
            outputs = self.upsample_layers[i](torch.cat([
                down_sample_outputs[- i - 2], 
                F.interpolate(outputs, size = interpolate_size, mode = "bilinear")
                ], dim = 1)
            )

        outputs = self.conv(F.interpolate(outputs, scale_factor = 2, mode = "bilinear"))
        return self.predict_kernel(input, outputs, white_level)


class PredictionKernelConv(nn.Module):
    def __init__(self):
        super(PredictionKernelConv, self).__init__()

    def forward(self, ori_input: Tensor, input: Tensor, white_level: Tensor) -> Tensor:
        burst_num = len(ori_input[0])
        K_power = len(input[0]) // burst_num

        output = torch.zeros_like(ori_input).to(input.device)
        for i in range(len(ori_input[0])):
            ori_frame = ori_input[:, i, :, :].unsqueeze(1)
            output[:, i, :, :] = torch.mean(ori_frame * input[:, K_power * i : K_power * (i + 1)], 1)
        output = output / white_level.squeeze(-1)
        
        if self.training:
            return torch.mean(output, 1), output
        else:
            return torch.mean(output, 1)
