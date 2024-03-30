import torch
import torchvision.transforms.functional
from torch import nn

class DoubleConvolution(nn.Module):
    # Two 3X3 Convolution layers
    '''
    Each step in the contraction path and expansive path have two 3X3 convolutional layers followed by ReLU activations
    In the original U-Net paper they used 0 padding, but I use 1 padding so that the final feature map in not cropped
    '''

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation1 = nn.ReLU()

        self.second = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.first(x)
        x = self.activation1(x)
        x = self.second(x)
        return self.activation2(x)
    

class DownSample(nn.Module):
    ''' 
    Each step in the contracting path down-samples the feature map with a 2X2 max pooling layer
    '''
    def __init__(self) -> None:
        super().__init__()

        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        return self.pool(x)

class UpSample(nn.Module):
    '''
    Eachstep in the expansive path up-samples the feature map with a 2X2 up-convolution
    '''
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x:torch.Tensor):
        return self.up(x)

# Work in Progress









