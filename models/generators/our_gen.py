"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F

from torch import nn
class UNET(nn.Module):
    def __init__(self, in_channels=4, out_channels=2):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

    def __call__(self, x):

        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                            )
        return expand

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.PReLU()
        )
        self.conv_1x1 = nn.Conv2d(in_features, in_features, kernel_size=1)

    def forward(self, x):
        return self.conv_1x1(x) + self.conv_block(x)


class ConvDownBlock(nn.Module):
    def __init__(self, in_chans, out_chans, batch_norm=True):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.batch_norm = batch_norm

        self.conv_1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        # self.conv_2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)
        self.res = ResidualBlock(out_chans)
        self.conv_3 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, stride=2)
        self.bn = nn.InstanceNorm2d(out_chans)
        self.activation = nn.PReLU()

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """

        if self.batch_norm:
            out = self.activation(self.bn(self.conv_1(input)))
            skip_out = self.res(out)  # self.activation(self.bn(self.conv_2(out)))
            out = self.conv_3(skip_out)
        else:
            out = self.activation(self.conv_1(input))
            skip_out = self.res(out)  # self.activation(self.conv_2(out))
            out = self.conv_3(skip_out)

        return out, skip_out


class ConvUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.conv_1 = nn.ConvTranspose2d(in_chans // 2, in_chans // 2, kernel_size=3, padding=1, stride=2)
        self.bn = nn.InstanceNorm2d(in_chans // 2)
        self.activation = nn.PReLU()

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.PReLU(),
            ResidualBlock(out_chans),
        )

    def forward(self, input, skip_input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """

        residual_skip = skip_input  # self.res_skip(skip_input)
        upsampled = self.activation(self.bn(self.conv_1(input, output_size=residual_skip.size())))
        concat_tensor = torch.cat([residual_skip, upsampled], dim=1)

        return self.layers(concat_tensor)


class GeneratorModel(nn.Module):
    def __init__(self, in_chans, out_chans, z_location=None, latent_size=1024):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.preprocess_unet = UNET()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = 256
        self.num_pool_layers = 4
        self.latent_size = latent_size

        num_pool_layers = self.num_pool_layers

        ch = self.chans

        self.down_sample_layers = nn.ModuleList([ConvDownBlock(in_chans, ch, batch_norm=False)])
        for i in range(num_pool_layers - 1):
            if i < 3:
                self.down_sample_layers += [ConvDownBlock(ch, ch * 2)]
                ch *= 2
            else:
                self.down_sample_layers += [ConvDownBlock(ch, ch)]

        self.res_layer_1 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(ch),
            nn.PReLU(),
            ResidualBlock(ch),
            ResidualBlock(ch),
            ResidualBlock(ch),
            ResidualBlock(ch),
            ResidualBlock(ch),
            # ResidualBlock(ch),
            # ResidualBlock(ch),
            # ResidualBlock(ch),
            # ResidualBlock(ch),
            # ResidualBlock(ch),
            # ResidualBlock(ch),
            # ResidualBlock(ch),
            # ResidualBlock(ch),
            # ResidualBlock(ch),
            # ResidualBlock(ch),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(ch),
            nn.PReLU(),
        )
        #
        # self.conv = nn.Sequential(
        #     nn.Conv2d(ch*2, ch, kernel_size=3, padding=1),
        #     nn.InstanceNorm2d(ch),
        #     nn.PReLU(),
        # )
        #
        # self.middle_z_grow_conv = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv2d(256, 1024, kernel_size=(3, 3), padding=1),
        #     nn.LeakyReLU(negative_slope=0.2),
        # )
        # self.middle_z_grow_linear = nn.Sequential(
        #     nn.Linear(512, 128 * 24 * 24),
        #     nn.LeakyReLU(negative_slope=0.2),
        # )

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            # if i > 0:
            self.up_sample_layers += [ConvUpBlock(ch * 2, ch // 2)]
            ch //= 2
            # else:
            #     self.up_sample_layers += [ConvUpBlock(ch * 2, ch)]


        self.up_sample_layers += [ConvUpBlock(ch * 2, ch)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
        )

    def forward(self, input, measured, z, mid_z=None):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        inp_noise = self.preprocess_unet(torch.cat([z, measured], dim=1))
        output = torch.cat([input, inp_noise], dim=1)
        stack = []
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output, skip_out = layer(output)
            stack.append(skip_out)

        # output = self.res_layer_1(output)
        # z_out = self.middle_z_grow_linear(mid_z)
        # z_out = torch.reshape(z_out, (output.shape[0], 128, 24, 24))
        # z_out = self.middle_z_grow_conv(z_out)
        # output = self.conv(torch.cat([output, z_out], dim=1))
        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = layer(output, stack.pop())

        return self.conv2(output)
