import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet_gb(nn.Module):
    def __init__(self, num_classes, activation):
        super(Unet_gb, self).__init__()
        Unet = torch.load('Unet_net.pth')

        enc1_block = list(Unet.enc1.block.children())
        for idx, f in enumerate(enc1_block):
            if isinstance(f, nn.ReLU):
                enc1_block.pop(idx)
                enc1_block.insert(idx, activation)

        enc1_block_out = list(Unet.enc1.block_out.children())
        for idx, f in enumerate(enc1_block_out):
            if isinstance(f, nn.ReLU):
                enc1_block_out.pop(idx)
                enc1_block_out.insert(idx, activation)

        enc2_block = list(Unet.enc2.block.children())
        for idx, f in enumerate(enc2_block):
            if isinstance(f, nn.ReLU):
                enc2_block.pop(idx)
                enc2_block.insert(idx, activation)

        enc2_block_out = list(Unet.enc2.block_out.children())
        for idx, f in enumerate(enc2_block_out):
            if isinstance(f, nn.ReLU):
                enc1_block_out.pop(idx)
                enc1_block_out.insert(idx, activation)

        enc3_block = list(Unet.enc3.block.children())
        for idx, f in enumerate(enc3_block):
            if isinstance(f, nn.ReLU):
                enc1_block.pop(idx)
                enc1_block.insert(idx, activation)

        enc3_block_out = list(Unet.enc3.block_out.children())
        for idx, f in enumerate(enc3_block_out):
            if isinstance(f, nn.ReLU):
                enc3_block_out.pop(idx)
                enc3_block_out.insert(idx, activation)

        enc4_block = list(Unet.enc4.block.children())
        for idx, f in enumerate(enc4_block):
            if isinstance(f, nn.ReLU):
                enc4_block.pop(idx)
                enc4_block.insert(idx, activation)

        enc4_block_out = list(Unet.enc4.block_out.children())
        for idx, f in enumerate(enc4_block_out):
            if isinstance(f, nn.ReLU):
                enc4_block_out.pop(idx)
                enc4_block_out.insert(idx, activation)

        center = list(Unet.center.children())
        for idx, f in enumerate(center):
            if isinstance(f, nn.ReLU):
                center.pop(idx)
                center.insert(idx, activation)

        dec1 = list(Unet.dec1.children())
        for idx, f in enumerate(dec1):
            if isinstance(f, nn.ReLU):
                dec1.pop(idx)
                dec1.insert(idx, activation)

        dec2_block = list(Unet.dec2.block.children())
        for idx, f in enumerate(dec2_block):
            if isinstance(f, nn.ReLU):
                dec2_block.pop(idx)
                dec2_block.insert(idx, activation)

        dec2_block_out = list(Unet.dec2.block_out.children())
        for idx, f in enumerate(dec2_block_out):
            if isinstance(f, nn.ReLU):
                dec2_block_out.pop(idx)
                dec2_block_out.insert(idx, activation)

        dec3_block = list(Unet.dec3.block.children())
        for idx, f in enumerate(dec3_block):
            if isinstance(f, nn.ReLU):
                dec3_block.pop(idx)
                dec3_block.insert(idx, activation)

        dec3_block_out = list(Unet.dec3.block_out.children())
        for idx, f in enumerate(dec3_block_out):
            if isinstance(f, nn.ReLU):
                dec3_block_out.pop(idx)
                dec3_block_out.insert(idx, activation)

        dec4_block = list(Unet.dec4.block.children())
        for idx, f in enumerate(dec4_block):
            if isinstance(f, nn.ReLU):
                dec4_block.pop(idx)
                dec4_block.insert(idx, activation)

        dec4_block_out = list(Unet.dec4.block_out.children())
        for idx, f in enumerate(dec4_block_out):
            if isinstance(f, nn.ReLU):
                dec4_block_out.pop(idx)
                dec4_block_out.insert(idx, activation)

        self.enc1_block = nn.Sequential(*enc1_block)
        self.enc1_block_out = nn.Sequential(*enc1_block_out)
        self.enc2_block = nn.Sequential(*enc2_block)
        self.enc2_block_out = nn.Sequential(*enc2_block_out)
        self.enc3_block = nn.Sequential(*enc3_block)
        self.enc3_block_out = nn.Sequential(*enc3_block_out)
        self.enc4_block = nn.Sequential(*enc4_block)
        self.enc4_block_out = nn.Sequential(*enc4_block_out)

        self.center = nn.Sequential(*center)

        self.dec1 = nn.Sequential(*dec1)
        self.dec2_block = nn.Sequential(*dec2_block)
        self.dec2_block_out = nn.Sequential(*dec2_block_out)
        self.dec3_block = nn.Sequential(*dec3_block)
        self.dec3_block_out = nn.Sequential(*dec3_block_out)
        self.dec4_block = nn.Sequential(*dec4_block)
        self.dec4_block_out = nn.Sequential(*dec4_block_out)

    def forward(self, x):                           # Forward pass for network.

        enc1_block = self.enc1_block(x)
        enc1_block_out = self.enc1_block_out(enc1_block)
        enc2_block = self.enc2_block(
                    F.max_pool2d(enc1_block_out, kernel_size=2, stride=2))
        enc2_block_out = self.enc2_block_out(enc2_block)
        enc3_block = self.enc3_block(
                    F.max_pool2d(enc2_block_out, kernel_size=2, stride=2))
        enc3_block_out = self.enc3_block_out(enc3_block)
        enc4_block = self.enc4_block(
                    F.max_pool2d(enc3_block_out, kernel_size=2, stride=2))
        enc4_block_out = self.enc4_block_out(enc4_block)

        center = self.center(
                    F.max_pool2d(enc4_block_out, kernel_size=2, stride=2))

        dec4_block = self.dec4_block(torch.cat([center, enc4_block_out], 1))
        dec4_block_out = self.dec4_block_out(dec4_block)
        dec3_block = self.dec3_block(
                    torch.cat([dec4_block_out, enc3_block_out], 1))
        dec3_block_out = self.dec3_block_out(dec3_block)
        dec2_block = self.dec2_block(
                    torch.cat([dec3_block_out, enc2_block_out], 1))
        dec2_block_out = self.dec2_block_out(dec2_block)
        dec1 = self.dec1(torch.cat([dec2_block_out, enc1_block_out], 1))

        out = dec1.permute(1, 0, 2, 3).contiguous()
        out = out.view(2, -1)
        out = out.permute(1, 0)

        return out
