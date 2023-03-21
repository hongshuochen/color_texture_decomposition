"""
 @Time    : 2023-03-21 07:29:40
 @Author  : Hong-Shuo Chen
 @E-mail  : hongshuo@usc.edu
 
 @Project : Camouflage Object Detection
 @File    : conv2d_pca.py
 @Function: Conv2d with PCA
"""
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union

# extract patches from image and reshape to (n*h*w, c*s*s)
def extract_patches_2d(x, kernel_size=(3,3), stride=(1,1), padding=(0,0), dilation=1, padding_mode='reflect'):
    # input: (n, c, h, w)
    # output: (n*h*w, c*s*s)
    if padding != 0:
        x = F.pad(x, (padding[1], padding[1], padding[0], padding[0]), padding_mode)
    n, c, h, w = x.shape
    if dilation == 1:
        # https://discuss.pytorch.org/t/how-to-extract-patches-from-an-image/79923/4
        patches = x.unfold(2, kernel_size[0], stride[0]).unfold(3, kernel_size[1], stride[1]) # (n, c, h, w, s, s)
        patches = patches.reshape(n, c, ((h-kernel_size[0])//stride[0]+1)*((w-kernel_size[1])//stride[1]+1), kernel_size[0], kernel_size[1]) # (n, c, h*w, s, s)
        patches = patches.permute(0,2,1,3,4) # (n, h*w, c, s, s)
        patches = patches.reshape(n*((h-kernel_size[0])//stride[0]+1)*((w-kernel_size[1])//stride[1]+1), c*kernel_size[0]*kernel_size[1]) # (n*h*w, c*s*s)
    else:
        # https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
        unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, stride=stride)
        patches = unfold(x) # (n, c*s*s, h*w)
        patches = patches.permute(0,2,1) # (n, h*w, c*s*s)
        patches = patches.reshape(-1,patches.shape[-1]) # (n*h*w, c*s*s)
    return patches

class Conv2d_PCA(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = 'reflect',
        device=None,
        dtype=None,
        mode='pca',
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.mode = mode
        
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)
        
    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            # print('training')
            if self.mode == 'pca':
                self._pca_init(input)
            elif self.mode == 'cwpca':
                self._cwpca_init(input)
        return self._conv_forward(input, self.weight, self.bias)
    
    # PCA initialization
    def _pca_init(self, input: Tensor) -> None:
        # get patches
        patches = extract_patches_2d(input, kernel_size=self.kernel_size, 
                                     stride=self.stride, padding=self.padding, 
                                     dilation=self.dilation, padding_mode=self.padding_mode)
        # run PCA
        _, S, V = torch.pca_lowrank(patches, q=patches.shape[1], center=True, niter=2)
        self.explained_variance_ratio_ = S/torch.sum(S)
        
        # set weight
        w = V.T.reshape(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        with torch.no_grad():
            self.weight.data = w
            
        self.training = False
            
    # channel-wise PCA initialization
    def _cwpca_init(self, input: Tensor) -> None:
        self.explained_variance_ratio_ = []
        # get patches
        for i in range(input.shape[1]):
            patches = extract_patches_2d(input[:,i:i+1], kernel_size=self.kernel_size, 
                                        stride=self.stride, padding=self.padding, 
                                        dilation=self.dilation, padding_mode=self.padding_mode)
            # run PCA
            _, S, V = torch.pca_lowrank(patches, q=patches.shape[1], center=True, niter=2)
            self.explained_variance_ratio_.append(S/torch.sum(S))
            
            # set weight
            w = V.T.reshape(patches.shape[1], 1, self.kernel_size[0], self.kernel_size[1])
            with torch.no_grad():
                self.weight.data[i*w.shape[0]:(i+1)*w.shape[0]] = w
                
        self.training = False