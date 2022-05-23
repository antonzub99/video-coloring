import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from dataset import VideoDataset


if __name__ == '__main__':
    data_root = ''
    frame_stack = 5

    dataset = VideoDataset(data_root, frame_stack, img_size=128)
    train_data = data.Subset(dataset, np.arange(len(dataset))[:-5])
    val_data = data.Subset(dataset, np.arange(len(dataset))[-5:])
    trainloader = data.DataLoader(train_data, num_workers=2, batch_size=4)
    valloader = data.DataLoader(val_data, num_workers=2, batch_size=1)

    batch = next(iter(trainloader))
    print(batch.shape)
