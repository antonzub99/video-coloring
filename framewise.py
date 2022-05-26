from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import colorization.colorizers as color


class FrameWiseNN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        # Merge time dimension with batch
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        # Apply the model
        x = self.model(x) 
        # Reshape and permute back
        _, C, H, W = x.shape
        return x.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)


def eccv16_framewise(pretrained=True):
    return FrameWiseNN(color.eccv16(pretrained))

def siggraph17_framewise(pretrained=True):
    return FrameWiseNN(color.siggraph17(pretrained))