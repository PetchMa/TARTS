"""Small U-Net for generating dynamic masks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block with batch normalization."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution with sigmoid activation for mask."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # return self.sigmoid(self.conv(x))
        return self.conv(x)


class SmallUNet(nn.Module):
    """Small U-Net for generating dynamic masks."""
    
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, 64)  # 32 -> 64
        self.down1 = Down(64, 128)  # 64 -> 128
        self.down2 = Down(128, 256)  # 128 -> 256
        self.down3 = Down(256, 512)  # 256 -> 512
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)  # 512 -> 1024
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)  # 1024 -> 512
        self.up2 = Up(512, 256 // factor, bilinear)  # 512 -> 256
        self.up3 = Up(256, 128 // factor, bilinear)  # 256 -> 128
        self.up4 = Up(128, 64, bilinear)  # 128 -> 64
        
        # Output
        self.outc = OutConv(64, n_classes)  # 64 -> n_classes
        
        # Initialize weights to prevent NaN
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights to prevent NaN values."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Check for NaN input
        if torch.isnan(x).any():
            print("Warning: NaN detected in U-Net input")
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output mask
        mask = self.outc(x)
        
        # Final NaN check
        if torch.isnan(mask).any():
            print("Warning: NaN detected in U-Net output, replacing with zeros")
            mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        
        return mask


class UNetMask(nn.Module):
    """U-Net based mask generator with optional temperature scaling."""
    
    def __init__(self, n_channels=1, temperature=1.0, hard_threshold=False, bilinear=True):
        super().__init__()
        self.unet = SmallUNet(n_channels=n_channels, n_classes=1, bilinear=bilinear)
        self.temperature = temperature
        self.hard_threshold = hard_threshold
        
    def forward(self, x):
        # Generate mask using U-Net
        mask = self.unet(x)
        
        # Apply temperature scaling for sharper masks
        # if self.temperature != 1.0:
        #     mask = torch.sigmoid(torch.logit(mask) / self.temperature)
        
        # # Optional hard thresholding
        # if self.hard_threshold:
        #     mask = (mask > 0.5).float()
            
        return mask
    
    def get_mask(self, x):
        """Get the generated mask without any post-processing.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W), (C, H, W), or (H, W)
            
        Returns
        -------
        torch.Tensor
            Generated mask of shape (B, 1, H, W) or (1, H, W)
        """
        # Handle different input dimensions
        if x.dim() == 2:
            # (H, W) -> (1, 1, H, W)
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            # (C, H, W) -> (1, C, H, W) -> (1, 1, H, W) if C > 1
            x = x.unsqueeze(0)  # Add batch dimension
            if x.shape[1] > 1:
                x = x[:, 0:1, :, :]  # Take only the first channel
        elif x.dim() == 4:
            # (B, C, H, W) -> (B, 1, H, W) if C > 1
            if x.shape[1] > 1:
                x = x[:, 0:1, :, :]  # Take only the first channel
        else:
            raise ValueError(f"Expected 2D, 3D, or 4D input, got {x.dim()}D")
        
        return self.unet(x)


if __name__ == "__main__":
    # Test the U-Net
    model = SmallUNet(n_channels=1, n_classes=1)
    x = torch.randn(1, 1, 160, 160)
    mask = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output mask shape: {mask.shape}")
    print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
    
    # Test UNetMask
    mask_model = UNetMask(n_channels=1, temperature=0.5)
    mask_output = mask_model(x)
    print(f"Mask output shape: {mask_output.shape}")
    print(f"Mask output range: [{mask_output.min():.3f}, {mask_output.max():.3f}]") 