"""
 @Time    : 2023-03-21 07:29:40
 @Author  : Hong-Shuo Chen
 @E-mail  : hongshuo@usc.edu
 
 @Project : Camouflage Object Detection
 @File    : color_texture_decomposition.py
 @Function: Color Textrue Decomposition
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from conv2d_pca import Conv2d_PCA
   
def gaussian_kernel(kernel_size=3, sigma=1):
    """
    creates gaussian kernel with side length `kernel_size` and a sigma of `sig`
    """
    ax = torch.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    gauss = torch.exp(-0.5 * torch.square(ax) / sigma**2)
    kernel = torch.outer(gauss, gauss)
    return kernel / torch.sum(kernel)

class Color_Texture_Decomposition(nn.Module):
    def __init__(self, average_size=7, sigma=1, mode="average"):
        super().__init__()
        self.pca = Conv2d_PCA(3, 3, kernel_size=1, mode='pca')
        
        self.conv = nn.Conv2d(3, 3, kernel_size=average_size, stride=1,
                              padding=average_size//2, dilation=1, 
                              groups=3, bias=False, padding_mode='reflect')
        with torch.no_grad():
            if mode == "average":
                self.conv.weight.data = torch.ones(self.conv.weight.data.size())/average_size**2
            elif mode == "gaussian":
                kernel = gaussian_kernel(average_size, sigma=sigma)
                self.conv.weight.data[0,0] = kernel
                self.conv.weight.data[1,0] = kernel
                self.conv.weight.data[2,0] = kernel

    def forward(self, x):
        mean = self.conv(x)
        delta = x - mean
        pqr = self.pca(delta)
        return mean, delta, pqr

class Color_Texture_Hop(nn.Module):
    def __init__(self, kernel_size=7, average_size=7, sigma=1, mode="average"):
        super().__init__()
        self.ctd = Color_Texture_Decomposition(average_size=average_size, sigma=sigma, mode=mode)
        self.conv_pqr = Conv2d_PCA(3, kernel_size**2*3, kernel_size=kernel_size, stride=1,
                                   padding=kernel_size//2, dilation=1, groups=3, bias=False, 
                                   padding_mode='reflect', mode='cwpca')

    def forward(self, x):
        mean, delta, pqr = self.ctd(x)
        hop1 = self.conv_pqr(pqr)
        return mean, hop1
    
class Multiscale_Color_Texture_Hop(nn.Module):
    def __init__(self, kernel_size=7, average_size=7, sigma=1, num_hops=4,
                 output_size=(224,224), interpolation_mode='bicubic', align_corners=True):
        super().__init__()
        self.output_size = output_size
        self.align_corners = align_corners
        self.interpolation_mode = interpolation_mode
        self.layers = []
        for i in range(num_hops):
            self.layers.append(Color_Texture_Hop(kernel_size=kernel_size, average_size=average_size, 
                                                 sigma=sigma, mode="average"))
        self.layers = nn.ModuleList(self.layers)
        
    def forward(self, x):
        output = []
        for i in range(len(self.layers)):
            if i == 0: # first layer
                input = x
            else:
                input = F.interpolate(x, size=(int(x.size()[2]/(2**i)), int(x.size()[3]/(2**i))), mode='bicubic', align_corners=True)
            features = self.layers[i](input)
            features = torch.cat(features, 1)
            # resize features to output size
            features = F.interpolate(features, size=self.output_size, mode=self.interpolation_mode, align_corners=self.align_corners)
            output.append(features) 
        return output