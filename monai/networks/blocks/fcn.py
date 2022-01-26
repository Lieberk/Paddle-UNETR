# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Type

import paddle
import paddle.nn as nn
from monai.networks.layers.factories import Act, Conv, Norm
from monai.utils import optional_import

models, _ = optional_import("torchvision", name="models")


class GCN(nn.Layer):
    """
    The Global Convolutional Network module using large 1D
    Kx1 and 1xK kernels to represent 2D kernels.
    """

    def __init__(self, inplanes: int, planes: int, ks: int = 7):
        """
        Args:
            inplanes: number of input channels.
            planes: number of output channels.
            ks: kernel size for one dimension. Defaults to 7.
        """
        super().__init__()

        conv2d_type: Type[nn.Conv2D] = Conv[Conv.CONV, 2]
        self.conv_l1 = conv2d_type(in_channels=inplanes, out_channels=planes, kernel_size=(ks, 1), padding=(ks // 2, 0))
        self.conv_l2 = conv2d_type(in_channels=planes, out_channels=planes, kernel_size=(1, ks), padding=(0, ks // 2))
        self.conv_r1 = conv2d_type(in_channels=inplanes, out_channels=planes, kernel_size=(1, ks), padding=(0, ks // 2))
        self.conv_r2 = conv2d_type(in_channels=planes, out_channels=planes, kernel_size=(ks, 1), padding=(ks // 2, 0))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            x: in shape (batch, inplanes, spatial_1, spatial_2).
        """
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class Refine(nn.Layer):
    """
    Simple residual block to refine the details of the activation maps.
    """

    def __init__(self, planes: int):
        """
        Args:
            planes: number of input channels.
        """
        super().__init__()

        relu_type: Type[nn.ReLU] = Act[Act.RELU]
        conv2d_type: Type[nn.Conv2D] = Conv[Conv.CONV, 2]
        norm2d_type: Type[nn.BatchNorm2D] = Norm[Norm.BATCH, 2]

        self.bn = norm2d_type(num_features=planes)
        self.relu = relu_type(True)
        self.conv1 = conv2d_type(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.conv2 = conv2d_type(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            x: in shape (batch, planes, spatial_1, spatial_2).
        """
        residual = x
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)

        return residual + x
