import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision.transforms as transforms
from dataset import VideoDataset
from full_model import VideoFlowColorizer, ColorDiscriminator


class Trainer:
    def __init__(self, args, **kwargs):
        self.data_root = kwargs['data_root']
        self.frame_stack = kwargs['frame_stack']
        self.batch_size = kwargs['batch_size']
        self.dataset = VideoDataset(self.data_root, self.frame_stack, img_size=kwargs['img_size'])
        train_data = data.Subset(self.dataset, np.arange(len(self.dataset))[:-5])
        val_data = data.Subset(self.dataset, np.arange(len(self.dataset))[-5:])
        self.trainloader = data.DataLoader(train_data, num_workers=2, batch_size=self.batch_size)
        self.valloader = data.DataLoader(val_data, num_workers=2, batch_size=1)


        self.lr_gen = kwargs['lr_gen']
        self.lr_disc = kwargs['lr_disc']
        self.wd = kwargs['wd']

        self.gen_model = VideoFlowColorizer(args, flow_ckpt=kwargs['flow_ckpt'],
                                            img_planes=313, fplanes=313)
        self.disc_model = ColorDiscriminator(in_size=3)

        self.gen_opt = torch.optim.Adam(self.gen_model.parameters(), lr=self.lr_gen, weight_decay=self.wd)
        self.disc_opt = torch.optim.Adam(self.disc_model.parameters(), lr=self.lr_disc, weight_decay=self.wd)

    def trainstep(self):
        pass



