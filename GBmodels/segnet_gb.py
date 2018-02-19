import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNet_gb(nn.Module):
    def __init__(self, num_classes, activation):
        super(SegNet_gb, self).__init__()
        Segnet = torch.load('Segnet_net.pth')

        enc1_block = list(Segnet.enc1.block.children())
        for idx, f in enumerate(enc1_block):
            if isinstance(f, nn.ReLU):
                enc1_block.pop(idx)
                enc1_block.insert(idx, activation)

        enc1_block_out = list(Segnet.enc1.block_out.children())
        for idx, f in enumerate(enc1_block_out):
            if isinstance(f, nn.ReLU):
                enc1_block_out.pop(idx)
                enc1_block_out.insert(idx, activation)

        enc2_block = list(Segnet.enc2.block.children())
        for idx, f in enumerate(enc2_block):
            if isinstance(f, nn.ReLU):
                enc2_block.pop(idx)
                enc2_block.insert(idx, activation)

        enc2_block_out = list(Segnet.enc2.block_out.children())
        for idx, f in enumerate(enc2_block_out):
            if isinstance(f, nn.ReLU):
                enc1_block_out.pop(idx)
                enc1_block_out.insert(idx, activation)

        enc3_block = list(Segnet.enc3.block.children())
        for idx, f in enumerate(enc3_block):
            if isinstance(f, nn.ReLU):
                enc3_block.pop(idx)
                enc3_block.insert(idx, activation)

        enc3_block_out = list(Segnet.enc3.block_out.children())
        for idx, f in enumerate(enc3_block_out):
            if isinstance(f, nn.ReLU):
                enc3_block_out.pop(idx)
                enc3_block_out.insert(idx, activation)

        enc4_block = list(Segnet.enc4.block.children())
        for idx, f in enumerate(enc4_block):
            if isinstance(f, nn.ReLU):
                enc4_block.pop(idx)
                enc4_block.insert(idx, activation)

        enc4_block_out = list(Segnet.enc4.block_out.children())
        for idx, f in enumerate(enc4_block_out):
            if isinstance(f, nn.ReLU):
                enc4_block_out.pop(idx)
                enc4_block_out.insert(idx, activation)

        enc5_block = list(Segnet.enc5.block.children())
        for idx, f in enumerate(enc5_block):
            if isinstance(f, nn.ReLU):
                enc5_block.pop(idx)
                enc5_block.insert(idx, activation)

        enc5_block_out = list(Segnet.enc5.block_out.children())
        for idx, f in enumerate(enc5_block_out):
            if isinstance(f, nn.ReLU):
                enc5_block_out.pop(idx)
                enc5_block_out.insert(idx, activation)

        dec1 = list(Segnet.dec1.children())
        for idx, f in enumerate(dec1):
            if isinstance(f, nn.ReLU):
                dec1.pop(idx)
                dec1.insert(idx, activation)

        dec2_block = list(Segnet.dec2.block.children())
        for idx, f in enumerate(dec2_block):
            if isinstance(f, nn.ReLU):
                dec2_block.pop(idx)
                dec2_block.insert(idx, activation)

        dec2_block_out = list(Segnet.dec2.block_out.children())
        for idx, f in enumerate(dec2_block_out):
            if isinstance(f, nn.ReLU):
                dec2_block_out.pop(idx)
                dec2_block_out.insert(idx, activation)

        dec3_block = list(Segnet.dec3.block.children())
        for idx, f in enumerate(dec3_block):
            if isinstance(f, nn.ReLU):
                dec3_block.pop(idx)
                dec3_block.insert(idx, activation)

        dec3_block_out = list(Segnet.dec3.block_out.children())
        for idx, f in enumerate(dec3_block_out):
            if isinstance(f, nn.ReLU):
                dec3_block_out.pop(idx)
                dec3_block_out.insert(idx, activation)

        dec4_block = list(Segnet.dec4.block.children())
        for idx, f in enumerate(dec4_block):
            if isinstance(f, nn.ReLU):
                dec4_block.pop(idx)
                dec4_block.insert(idx, activation)

        dec4_block_out = list(Segnet.dec4.block_out.children())
        for idx, f in enumerate(dec4_block_out):
            if isinstance(f, nn.ReLU):
                dec4_block_out.pop(idx)
                dec4_block_out.insert(idx, activation)

        dec5_block = list(Segnet.dec5.block.children())
        for idx, f in enumerate(dec5_block):
            if isinstance(f, nn.ReLU):
                dec5_block.pop(idx)
                dec5_block.insert(idx, activation)

        dec5_block_out = list(Segnet.dec5.block_out.children())
        for idx, f in enumerate(dec5_block_out):
            if isinstance(f, nn.ReLU):
                dec5_block_out.pop(idx)
                dec5_block_out.insert(idx, activation)

        self.enc1_block = nn.Sequential(*enc1_block)
        self.enc1_block_out = nn.Sequential(*enc1_block_out)
        self.enc2_block = nn.Sequential(*enc2_block)
        self.enc2_block_out = nn.Sequential(*enc2_block_out)
        self.enc3_block = nn.Sequential(*enc3_block)
        self.enc3_block_out = nn.Sequential(*enc3_block_out)
        self.enc4_block = nn.Sequential(*enc4_block)
        self.enc4_block_out = nn.Sequential(*enc4_block_out)
        self.enc5_block = nn.Sequential(*enc5_block)
        self.enc5_block_out = nn.Sequential(*enc5_block_out)

        self.dec1 = nn.Sequential(*dec1)

        self.dec2_block = nn.Sequential(*dec2_block)
        self.dec2_block_out = nn.Sequential(*dec2_block_out)
        self.dec3_block = nn.Sequential(*dec3_block)
        self.dec3_block_out = nn.Sequential(*dec3_block_out)
        self.dec4_block = nn.Sequential(*dec4_block)
        self.dec4_block_out = nn.Sequential(*dec4_block_out)
        self.dec5_block = nn.Sequential(*dec5_block)
        self.dec5_block_out = nn.Sequential(*dec5_block_out)

    def forward(self, x):                           # Forward pass for network.

        enc1_block = self.enc1_block(x)
        enc1_block_out, idx1 = F.max_pool2d(
                self.enc1_block_out(enc1_block), 2, 2, return_indices=True)
        enc2_block = self.enc2_block(enc1_block_out)
        enc2_block_out, idx2 = F.max_pool2d(
                self.enc2_block_out(enc2_block), 2, 2, return_indices=True)
        enc3_block = self.enc3_block(F.dropout2d(enc2_block_out))
        enc3_block_out, idx3 = F.max_pool2d(
                self.enc3_block_out(enc3_block), 2, 2, return_indices=True)
        enc4_block = self.enc4_block(F.dropout2d(enc3_block_out))
        enc4_block_out, idx4 = F.max_pool2d(
                self.enc4_block_out(enc4_block), 2, 2, return_indices=True)
        enc5_block = self.enc5_block(F.dropout2d(enc4_block_out))
        enc5_block_out, idx5 = F.max_pool2d(
                self.enc5_block_out(enc5_block), 2, 2, return_indices=True)

        dec5_block = self.dec5_block(F.dropout2d(
                F.max_unpool2d(enc5_block_out, idx5, 2, 2)))
        dec5_block_out = self.dec5_block_out(dec5_block)
        dec4_block = self.dec4_block(F.dropout2d(
                F.max_unpool2d(dec5_block_out, idx4, 2, 2)))
        dec4_block_out = self.dec4_block_out(dec4_block)
        dec3_block = self.dec3_block(F.dropout2d(
                F.max_unpool2d(dec4_block_out, idx3, 2, 2)))
        dec3_block_out = self.dec3_block_out(dec3_block)
        dec2_block = self.dec2_block(
                F.max_unpool2d(dec3_block_out, idx2, 2, 2))
        dec2_block_out = self.dec2_block_out(dec2_block)
        dec1 = self.dec1(F.max_unpool2d(dec2_block_out, idx1, 2, 2))

        out = dec1.permute(1, 0, 2, 3).contiguous()
        out = out.view(2, -1)
        out = out.permute(1, 0)

        return out
