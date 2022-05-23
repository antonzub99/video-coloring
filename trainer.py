import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from torchvision import io
from torchvision import utils as visionutils

import wandb
from skimage import color
import cv2

from dataset import VideoDataset
from full_model import VideoFlowColorizer, ColorDiscriminator
from losses import AnchorConsistencyLoss, PerceptualLoss


def combineLab(tensorL, tensorAB, mode='bilinear'):
    HW_orig = tensorL.shape[-2:]
    HW = tensorAB.shape[-2:]

    if HW_orig[0] != HW[0] or HW_orig[1] != HW[1]:
        out_ab_orig = F.interpolate(tensorAB, size=HW_orig, mode=mode)
    else:
        out_ab_orig = tensorAB
    out_ab_orig = torch.cat((tensorL, out_ab_orig), dim=1)
    return color.lab2rgb(out_ab_orig.permute(0, 2, 3, 4, 1))


class Trainer:
    def __init__(self, args, **kwargs):
        self.device = torch.device("cuda:0") if torch.cuda.is_available else torch.device("cpu")
        self.logger = kwargs['logger']

        self.data_root = kwargs['data_root']
        self.frame_stack = kwargs['frame_stack']
        self.batch_size = kwargs['batch_size']

        self.max_iters = kwargs['max_iters']
        self.validate_every = kwargs['val_rate']
        self.save_every = kwargs['save_rate']
        self.models_ckpt = kwargs['ckpt_root']

        self.val_root = kwargs['val_root']
        self.dataset = VideoDataset(self.data_root, self.frame_stack, img_size=kwargs['img_size'])
        train_data = data.Subset(self.dataset, np.arange(len(self.dataset))[:-5])
        val_data = data.Subset(self.dataset, np.arange(len(self.dataset))[-5:])
        self.trainloader = data.DataLoader(train_data, num_workers=2, batch_size=self.batch_size)
        self.valloader = data.DataLoader(val_data, num_workers=2, batch_size=1)

        self.lr_gen = kwargs['lr_gen']
        self.lr_disc = kwargs['lr_disc']
        self.wd = kwargs['wd']

        self.gen_model = VideoFlowColorizer(args, flow_ckpt=kwargs['flow_ckpt'],
                                            img_planes=313, fplanes=313).to(self.device)
        self.disc_model = ColorDiscriminator(in_size=3).to(self.device)

        self.gen_opt = torch.optim.Adam(self.gen_model.parameters(), lr=5 * self.lr_gen, weight_decay=self.wd)
        self.disc_opt = torch.optim.Adam(self.disc_model.parameters(), lr=self.lr_disc, weight_decay=self.wd)

        self.perceptual_loss, self.anchor_loss, self.disc_loss = self.init_losses()
        self.lambda_anchor = kwargs['lambda_anchor']

    def init_losses(self):
        perceptual = PerceptualLoss().to(self.device)
        anchor = AnchorConsistencyLoss()
        disc_loss = nn.BCEWithLogitsLoss()
        return perceptual, anchor, disc_loss

    def trainstep(self):
        frames, target, _ = next(iter(self.trainloader))
        # frames have shape (B x C x T x H x W)
        # frames have 1 channel (L), target 2 (ab)
        self.gen_model.train()
        self.disc_model.train()

        frames = frames.to(self.device)
        target = target.to(self.device)

        ab_pred = self.gen_model(frames)
        lab_pred = torch.cat((frames, ab_pred), dim=1)
        lab_true = torch.cat((frames, target), dim=1)
        b, ch, t, h, w = lab_pred.shape

        # reshape for conv2d processing in perceptual loss
        perceptual = self.perceptual_loss(lab_pred.view(b * t, ch, h, w), lab_true.view(b * t, ch, h, w))
        anchor = self.anchor_loss(ab_pred[:, :, 0, :, :], ab_pred[:, :, -1, :, :])
        disc_pred = self.disc_model(lab_pred.view(b * t, ch, h, w)
        disc_score = self.disc_loss(disc_pred, torch.ones_like(disc_pred))

        gen_loss = disc_score + perceptual + self.lambda_anchor * anchor

        gen_loss.backward()
        self.gen_opt.step()
        self.disc_opt.step()
        self.gen_model.zero_grad()
        self.disc_model.zero_grad()

        if self.logger:
            wandb.log({f'Train/Loss/perceptual': perceptual.item(),
                       f'Train/Loss/anchor_consistency': anchor.item()})
        # discriminator also takes Lab images in

        lab_pred = torch.cat((frames, ab_pred), dim=1)
        lab_true = torch.cat((frames, target), dim=1)
        b, ch, t, h, w = lab_pred.shape

        lab_pred = lab_pred.view(b * t, ch, h, w)
        lab_true = lab_true.view(b * t, ch, h, w)
        disc_pred = self.disc_model(lab_pred.detach())
        disc_true = self.disc_model(lab_true)

        disc_loss = self.disc_loss(disc_pred, torch.zeros_like(disc_pred)) + \
            self.disc_loss(disc_true, torch.ones_like(disc_true))

        disc_loss.backward()
        self.gen_opt.step()
        self.disc_opt.step()
        self.gen_model.zero_grad()
        self.disc_model.zero_grad()

        if self.logger:
            wandb.log({'Train/Loss/Discriminator': disc_loss.item()})

    def valstep(self, cur_idx):
        frames, target, _ = next(iter(self.valloader))
        self.gen_model.eval()

        frames = frames.to(self.device)
        target = target.to(self.device)

        with torch.no_grad():
            ab_pred = self.gen_model(frames)
            lab_pred = torch.cat((frames, ab_pred), dim=1)
            lab_true = torch.cat((frames, target), dim=1)

            b, ch, t, h, w = lab_pred.shape
            perceptual = self.perceptual_loss(lab_pred.view(b * t, ch, h, w), lab_true.view(b * t, ch, h, w))
            anchor = self.anchor_loss(ab_pred[:, :, 0, :, :], ab_pred[:, :, -1, :, :])
            gen_loss = perceptual + self.lambda_anchor * anchor

        if self.logger:
            wandb.log({f'Val/Loss/perceptual': perceptual.item(),
                       f'Val/Loss/anchor': anchor.item()})

        rgb_pred = torch.from_numpy(combineLab(frames.cpu(), ab_pred.cpu())).permute(0, 4, 1, 2, 3)
        rgb_true = torch.from_numpy(combineLab(frames.cpu(), target.cpu())).permute(0, 4, 1, 2, 3)

        video = []
        for idx in range(t):
            # concat along spatial dim
            batched = torch.cat((rgb_pred[:, :, idx, :, :], rgb_true[:, :, idx, :, :]), dim=2)
            batched = visionutils.make_grid(batched.data.detach().cpu(), nrow=1)
            video.append(batched.permute(1, 2, 0))

        h, w, ch = video[0].shape
        if not os.path.isdir(f'{self.val_root}'):
            os.makedirs(f'{self.val_root}')

        video_name = f'{self.val_root}/video{cur_idx:08d}.mp4'
        #io.write_video(video_name, (255 * torch.cat(video, dim=0).clamp(0, 1)).to(torch.uint8), fps=1, video_codec='h264')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_name, fourcc, 1, (w, h))
        if not video_writer.isOpened():
            print("Video writing error")

        for frame in video:
            if not video_writer.isOpened():
                print("Frame writing error")
            np_frame = (np.clip(frame.numpy(), 0, 1) * 255).astype(np.uint8)
            video_writer.write(np_frame)
        video_writer.release()

        if self.logger:
            wandb.log({"Video/Val": wandb.Video(video_name)})

    def train(self, exp_name, cfg):
        ckpt_folder = f'{self.models_ckpt}/checkpoints'
        if not os.path.isdir(ckpt_folder):
            os.makedirs(ckpt_folder)

        if self.logger:
            wandb.init(
                project='DL-proj',
                entity='antonzub',
                name=exp_name,
                config=cfg
            )

        for step_idx in range(self.max_iters):
            self.trainstep()

            if step_idx % self.validate_every == 0:
                self.valstep(step_idx)

            if step_idx % self.save_every == 0:
                torch.save({
                    'generator': self.gen_model.state_dict(),
                    'discriminator': self.disc_model.state_dict()
                }, f=ckpt_folder+'/checkpoint.pth')

        wandb.finish()
        return self.gen_model, self.disc_model
