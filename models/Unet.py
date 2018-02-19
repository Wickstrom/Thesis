import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

#############################################################################
# Implementation of U-net from https://arxiv.org/abs/1505.04597.            #
#############################################################################


class Encoder(nn.Module):
    def __init__(self, n_in, n_out, activation):
        super(Encoder, self).__init__()

        # Encoder consisting of conv layer -> batch normalization ->
        # activation function.

        # n_in: Number of features in to encoder.
        # n_out: Number of features out of encoder
        # activation: Activation function.

        self.block = nn.Sequential(
            *([nn.Conv2d(n_in, n_out, kernel_size=3, padding=1),
               nn.BatchNorm2d(n_out),
               activation, ])
        )
        self.block_out = nn.Sequential(
            *([nn.Conv2d(n_out, n_out, kernel_size=3, padding=1),
               nn.BatchNorm2d(n_out),
               activation, ])
        )

    def forward(self, x):                           # Forward pass for encoder.

        out = self.block(x)
        out = self.block_out(out)
        return(out)


class Decoder(nn.Module):
    def __init__(self, n_in, n_mid, n_out, activation):
        super(Decoder, self).__init__()

        # Decoder consisting of conv layer -> batch normalization ->
        # activation function -> transposed convolution.

        # n_in: Number of features in to decoder.
        # n_mid: Number of features in center part of decoder.
        # n_out: Number of features out of decoder.
        # activation: Activation function.

        self.block = nn.Sequential(
            *([nn.Conv2d(n_in, n_mid, kernel_size=3, padding=1),
               nn.BatchNorm2d(n_mid),
               activation, ])
        )
        self.block_out = nn.Sequential(
            *([nn.Conv2d(n_mid, n_out, kernel_size=3, padding=1),
               nn.BatchNorm2d(n_out),
               activation,
               nn.ConvTranspose2d(n_out, n_out, kernel_size=4,
                                  stride=2, padding=1, bias=False), ])
        )

    def forward(self, x):                           # Forward pass for decoder.

        out = self.block(x)
        out = self.block_out(out)

        return(out)


class Unet(nn.Module):
    def __init__(self, num_classes, activation):
        super(Unet, self).__init__()

        # U-net with dropout included in the center block, dropout rate = 0.5.
        # Max-pooling is performed in the forward pass function.
        # Last part of forward pass reshapes output into shape
        # (Number of pixels * Number of images, Number of classe ) to fit
        # cross entropy cost of Pytorch.

        self.enc1 = Encoder(3, 64, activation)
        self.enc2 = Encoder(64, 128, activation)
        self.enc3 = Encoder(128, 256, activation)
        self.enc4 = Encoder(256, 512, activation)

        self.center = nn.Sequential(
            *([nn.Dropout2d(),
               nn.Conv2d(512, 1024, kernel_size=3, padding=1),
               nn.BatchNorm2d(1024),
               activation,
               nn.Dropout2d(),
               nn.Conv2d(1024, 512, kernel_size=3, padding=1),
               nn.BatchNorm2d(512),
               activation,
               nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2,
                                  padding=1), ])
        )

        self.dec4 = Decoder(1024, 512, 256, activation)
        self.dec3 = Decoder(512, 256, 128, activation)
        self.dec2 = Decoder(256, 128, 64, activation)
        self.dec1 = nn.Sequential(
            *([nn.Conv2d(128, 64, kernel_size=3, padding=1),
               nn.BatchNorm2d(64),
               activation,
               nn.Conv2d(64, 64, kernel_size=3, padding=1),
               nn.BatchNorm2d(64),
               activation,
               nn.Conv2d(64, num_classes, kernel_size=1, padding=0), ])
        )

        for m in self.modules():
            self.weight_init(m)

    def forward(self, x):                           # Forward pass for network.

        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2, stride=2))

        center = self.center(F.max_pool2d(enc4, kernel_size=2, stride=2))

        dec4 = self.dec4(torch.cat([center, enc4], 1))
        dec3 = self.dec3(torch.cat([dec4, enc3], 1))
        dec2 = self.dec2(torch.cat([dec3, enc2], 1))
        dec1 = self.dec1(torch.cat([dec2, enc1], 1))

        out = dec1.permute(1, 0, 2, 3).contiguous()
        out = out.view(2, -1)
        out = out.permute(1, 0)

        return out

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight.data)
            init.constant(m.bias.data, 1)
        if isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight.data, 1)
            init.constant(m.bias.data, 0)
