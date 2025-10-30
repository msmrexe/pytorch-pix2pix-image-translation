"""
Defines the core model architectures:
- DownSample (Encoder block)
- UpSample (Decoder block)
- Generator (U-Net)
- Discriminator (PatchGAN)
"""
import torch
import torch.nn as nn

class DownSample(nn.Module):
    """
    A single downsampling block for the U-Net Encoder.
    Conv -> (BatchNorm) -> LeakyReLU
    """
    def __init__(self, in_channels, out_channels, apply_batchnorm=True):
        super(DownSample, self).__init__()
        
        layers = [
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=not(apply_batchnorm) # No bias if using BatchNorm
            )
        ]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)

class UpSample(nn.Module):
    """
    A single upsampling block for the U-Net Decoder.
    ConvTranspose -> BatchNorm -> (Dropout) -> ReLU
    """
    def __init__(self, in_channels, out_channels, apply_dropout=False):
        super(UpSample, self).__init__()
        
        layers = [
            nn.ConvTranspose2d(
                in_channels, 
                out_channels, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False # Using BatchNorm
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if apply_dropout:
            layers.append(nn.Dropout(0.5))
            
        self.up = nn.Sequential(*layers)

    def forward(self, x, skip_connection):
        """
        Forward pass with skip connection concatenation.
        """
        x = self.up(x)
        x = torch.cat([x, skip_connection], dim=1) # Concatenate along channel dim
        return x

class Generator(nn.Module):
    """
    The U-Net Generator architecture for Pix2Pix.
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()
        
        # Encoder (Downsampling)
        self.down1 = DownSample(in_channels, 64, apply_batchnorm=False) # C64
        self.down2 = DownSample(64, 128)   # C128
        self.down3 = DownSample(128, 256)  # C256
        self.down4 = DownSample(256, 512)  # C512
        self.down5 = DownSample(512, 512)  # C512
        self.down6 = DownSample(512, 512)  # C512
        self.down7 = DownSample(512, 512)  # C512
        
        # Bottleneck
        self.bottleneck = DownSample(512, 512) # C512

        # Decoder (Upsampling)
        # Input channels are doubled due to skip connections
        self.up1 = UpSample(512, 512, apply_dropout=True)   # CD512
        self.up2 = UpSample(1024, 512, apply_dropout=True)  # CD1024
        self.up3 = UpSample(1024, 512, apply_dropout=True)  # CD1024
        self.up4 = UpSample(1024, 512) # C1024
        self.up5 = UpSample(1024, 256) # C1024
        self.up6 = UpSample(512, 128)  # C512
        self.up7 = UpSample(256, 64)   # C256
        
        # Final Layer
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                128, # From up7 + down1
                out_channels, 
                kernel_size=4, 
                stride=2, 
                padding=1
            ),
            nn.Tanh() # Output in range [-1, 1]
        )

    def forward(self, x):
        # Encoder forward pass
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        
        b = self.bottleneck(d7) # Bottleneck

        # Decoder forward pass with skip connections
        u1 = self.up1(b, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        
        return self.final_up(u7)


class Discriminator(nn.Module):
    """
    The PatchGAN Discriminator architecture.
    """
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        
        # Helper block
        def conv_block(in_c, out_c, stride):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        # Input is concatenated (input_img + target_img), so channels = in_channels * 2
        self.model = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            conv_block(64, 128, stride=2),   # C128
            conv_block(128, 256, stride=2),  # C256
            
            # Stride 1 for 70x70 PatchGAN receptive field (as per paper)
            conv_block(256, 512, stride=1),  # C512 
            
            # Final output layer
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
            # No Sigmoid here, as we use BCEWithLogitsLoss for stability
        )

    def forward(self, x, y):
        """
        Forward pass.
        Args:
            x: Input image (e.g., segmented map)
            y: Target/Generated image (e.g., real photo)
        """
        concatenated = torch.cat([x, y], dim=1)
        verdict = self.model(concatenated)
        return verdict
