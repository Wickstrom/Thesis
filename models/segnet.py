import torch.nn as nn
import torch.nn.functional as F

#############################################################################
# Implementation of Segnet from https://arxiv.org/abs/1511.00561            #
#############################################################################


class Encoder(nn.Module):
    def __init__(self, n_in, n_out, n_l, activation):
        super(Encoder, self).__init__()

        # Encoder consisting of conv layer -> batch normalization ->
        # activation function.

        # n_in: Number of features in to encoder.
        # n_out: Number of features out of encoder
        # n_l: Number of layers in a block.
        # activation: Activation function.

        self.block = nn.Sequential(
            *([nn.Conv2d(n_in, n_out, kernel_size=3, padding=1),
               nn.BatchNorm2d(n_out),
               activation, ])
        )
        self.block_out = nn.Sequential(
            *([nn.Conv2d(n_out, n_out, kernel_size=3, padding=1),
               nn.BatchNorm2d(n_out),
               activation, ]*n_l)
        )
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, x):                           # Forward pass for encoder.

        out = self.block(x)
        out = self.block_out(out)
        out, idx = self.pool(out)

        return out, idx


class Decoder(nn.Module):
    def __init__(self, n_in, n_out, n_l, activation):
        super(Decoder, self).__init__()

        # Decoder consisting of conv layer -> batch normalization ->
        # activation function -> Max unpooling with indices from
        # encoder.

        # n_in: Number of features in to encoder.
        # n_out: Number of features out of encoder
        # n_l: Number of layers in a block.
        # activation: Activation function.

        self.block = nn.Sequential(
            *([nn.Conv2d(n_in, n_in, kernel_size=3, padding=1),
               nn.BatchNorm2d(n_in),
               activation, ] * n_l)
        )
        self.block_out = nn.Sequential(
            *([nn.Conv2d(n_in, n_out, kernel_size=3, padding=1),
               nn.BatchNorm2d(n_out),
               activation, ])
        )

        self.unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, x, idx):                      # Forward pass for decoder.

        out = self.unpool(x, idx)
        out = self.block(out)
        out = self.block_out(out)

        return out


class SegNet(nn.Module):
    def __init__(self, num_classes, activation):
        super(SegNet, self).__init__()

        # Segnet with dropout included in the last three
        # encoder at three first decoders with dropout rate = 0.5.
        # Max-pooling is performed in the forward pass function.
        # Last part of forward pass reshapes output into shape
        # (Number of pixels * Number of images, Number of classe ) to fit
        # cross entropy cost of Pytorch.

        self.enc1 = Encoder(3, 64, 1, activation)
        self.enc2 = Encoder(64, 128, 1, activation)
        self.enc3 = Encoder(128, 256, 2, activation)
        self.enc4 = Encoder(256, 512, 2, activation)
        self.enc5 = Encoder(512, 512, 2, activation)

        self.dec5 = Decoder(512, 512, 2, activation)
        self.dec4 = Decoder(512, 256, 2, activation)
        self.dec3 = Decoder(256, 128, 2, activation)
        self.dec2 = Decoder(128, 64, 1, activation)
        self.dec1 = nn.Sequential(
            *([
               nn.Conv2d(64, 64, kernel_size=3, padding=1),
               nn.BatchNorm2d(64),
               activation,
               nn.Conv2d(64, num_classes, kernel_size=3, padding=1), ])
            )

    def forward(self, x):                           # Forward pass for network.

        enc1, idx1 = self.enc1(x)
        enc2, idx2 = self.enc2(enc1)
        enc3, idx3 = self.enc3(F.dropout2d(enc2))
        enc4, idx4 = self.enc4(F.dropout2d(enc3))
        enc5, idx5 = self.enc5(F.dropout2d(enc4))

        dec5 = self.dec5(F.dropout2d(enc5), idx5)
        dec4 = self.dec4(F.dropout2d(dec5), idx4)
        dec3 = self.dec3(F.dropout2d(dec4), idx3)
        dec2 = self.dec2(dec3, idx2)
        dec1 = self.dec1(F.max_unpool2d(dec2, idx1, 2, 2))

        out = dec1.permute(1, 0, 2, 3).contiguous()
        out = out.view(2, -1)
        out = out.permute(1, 0)

        return out
