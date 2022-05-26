import utils
import remasternet
import loss
import torch
from torchvision.transforms import Resize
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

class FullVideoDataset(Dataset):
    def __init__(self, root, reference_size=10, img_size=128, extensions=None, transform=None, target_transform=None):
        self.root = root
        self.img_size = img_size
        self.reference_size = reference_size
        self.loader = self.default_loader
        self.extensions = extensions
        self.transform = transform
        self.target_transform = target_transform

        classes, class_to_idx = self.find_classes(self.root)
        self.samples = self.make_dataset(root, class_to_idx)

    def make_dataset(self, directory, class_to_idx):
        samples = []

        for target_class in tqdm(sorted(class_to_idx.keys())):
            # print(target_class)
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue

            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                num_files = len(fnames)
                sorted_fnames = sorted(fnames)
                frames = []
                targets = []
                for i in range(num_files):
                    path = os.path.join(root, sorted_fnames[i])
                    img = self.loader(path).resize((self.img_size, self.img_size))
                    # img = Resize((self.img_size, self.img_size))(self.loader(path))
                    image_l, ab = utils.convertRGB2LABTensor(img)
                    frames.append(image_l)
                    targets.append(ab)

                sample = torch.stack(frames, dim=1)
                target = torch.stack(targets, dim=1)
                samples.append((sample, target))

        return samples

    def __getitem__(self, index):
        frames, targets = self.samples[index]
        if self.transform is not None:
            frames = self.transform(frames)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        reference_idx = np.random.randint(0, frames.shape[1], size=self.reference_size)
        reference = torch.cat((frames[:, reference_idx, ...], targets[:, reference_idx, ...]), axis=0)

        return frames, targets, reference

    @staticmethod
    def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def pil_loader(self, path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def default_loader(self, path: str) -> Any:
        return self.pil_loader(path)

    def __len__(self):
        return len(self.samples)

class VideoDataset(Dataset):
    def __init__(self, root, frame_stack, img_size=128, max_videos=-1,
                 reference_size=5, extensions=None, transform=None,
                 target_transform=None, reference_transform=None):
        self._flag_transform = True
        self.max_videos = max_videos
        self.root = root
        self.reference_size = reference_size
        self.frame_stack = frame_stack
        self.img_size = img_size
        self.loader = self.default_loader
        self.extensions = extensions
        self.transform = transform
        self.target_transform = target_transform
        self.reference_transform = reference_transform

        classes, class_to_idx = self.find_classes(self.root)
        self.samples = self.make_dataset(root, class_to_idx, frame_stack)

    def make_dataset(self, directory, class_to_idx, frame_stack):
        samples = []

        for target_class in tqdm(sorted(class_to_idx.keys())):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue

            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                num_files = len(fnames)
                sorted_fnames = sorted(fnames)
                frame_idx = 0
                while num_files - frame_idx > frame_stack:
                    frames = []
                    targets = []
                    for i in range(frame_stack):
                        path = os.path.join(root, sorted_fnames[frame_idx + i])
                        img = Resize((self.img_size, self.img_size))(self.loader(path))
                        # img = Resize((self.img_size, self.img_size))(self.loader(path))
                        image_l, ab = utils.convertRGB2LABTensor(img)
                        frames.append(image_l)
                        targets.append(ab)

                    sample = torch.stack(frames, dim=1)
                    target = torch.stack(targets, dim=1)
                    samples.append((sample, target))

                    frame_idx += frame_stack

        return samples

    def __getitem__(self, index):
        frames, targets = self.samples[index]
        frame_idx = np.random.randint(0, self.frame_stack)
        samples_idx = np.random.randint(0, len(self.samples), size=self.reference_size - 1)
        references = [torch.cat((self.samples[sample_idx][0][:, frame_idx, ...], self.samples[sample_idx][1][:, frame_idx, ...]), dim=0) for sample_idx in samples_idx]
        references.append(torch.cat((frames[:, frame_idx, ...], targets[:, frame_idx, ...]), dim=0))
        references = torch.stack(references, dim=1)

        assert references.shape[1] == self.reference_size

        if self.reference_transform is not None and self._flag_transform:
            references = self.reference_transform(references)

        if self.transform is not None and self._flag_transform:
            frames, targets = self.transform(frames, targets)
        # reference = torch.cat((frames[:, reference_idx, ...], targets[:, reference_idx, ...]), dim=0)

        return frames, targets, references

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        classes = classes[:self.max_videos]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def pil_loader(self, path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def default_loader(self, path: str) -> Any:
        return self.pil_loader(path)

    def __len__(self):
        return len(self.samples)

class ColorModel(pl.LightningModule):
    def __init__(
        self,
        train_dataset,
        test_dataset,
        optimizer: str = 'default',
        scheduler: str = 'default',
        lr: float = None,
        wd : float = None,
        batch_size: int = 16,
        plot_images: int = 5
    ):
        super().__init__()
        self.net = remasternet.NetworkC()

        # dataset_len = len(dataset)
        # train_size = int(dataset_len * 0.7)
        # test_size = dataset_len - train_size
        # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_loss = loss.TrainLoss()
        self.val_loss = loss.ValLoss()

        self.plot_images = plot_images
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.wd = wd
        self.eps = 1e-7

    def forward(self, x, reference):
        # output_l = self.restor(x)
        return self.net(x, reference)

    def training_step(self, batch, batch_idx):
        lum, ab, reference  = batch

        ab_pred = self.forward(lum, reference)
        # l_pred = self.restor(lum)
        # ab_pred = self.net(l_pred, reference)
        # train_loss = self.train_loss(l_pred, ab_pred, lum, ab)

        # TODO: try focal loss
        train_loss = self.train_loss(ab_pred, ab)

        self.log('train_loss', train_loss, prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        lum, ab, reference = batch
        pred_ab = self.forward(lum, reference)
        reference_idx = 0
        lab_preds = torch.cat((lum[:self.plot_images, :, reference_idx, ...], pred_ab[:self.plot_images, :, reference_idx, ...]), dim=1).permute(0,2,3,1).cpu().numpy()
        lab_reals = torch.cat((lum[:self.plot_images, :, reference_idx, ...], ab[:self.plot_images, :, reference_idx, ...]), dim=1).permute(0,2,3,1).cpu().numpy()
        # lab_reals = torch.cat((lum[:, :, reference_idx, ...], ab[:, :, reference_idx, ...]), dim=1).permute(0,2,3,4,1).cpu().numpy()
        img_preds = torch.from_numpy(utils.convertLAB2RGB(lab_preds)).permute(0,3,1,2).to(ab.device)
        img_reals = torch.from_numpy(utils.convertLAB2RGB(lab_reals)).permute(0,3,1,2).to(ab.device) # (B, T, C, H, W)

        l1_ab = self.train_loss(pred_ab, ab)
        l1_rgb, lpips, psnr = self.val_loss(img_preds.float(), img_reals.float())

        return {'test_loss': l1_ab, 
                'l1_rgb': l1_rgb, 
                'lpips': lpips,
                'psnr' : psnr,
                'img_pred': img_preds, 
                'img_real': img_reals, 
            }
 
    def validation_epoch_end(self, outputs):
        test_loss = np.mean([x['test_loss'].cpu().item() for x in outputs])
        l1_rgb = np.mean([x['l1_rgb'].cpu().item() for x in outputs])
        lpips = np.mean([x['lpips'].cpu().item() for x in outputs])
        psnr = np.mean([x['psnr'].cpu().item() for x in outputs])
 
        log_dict = {'test_loss': test_loss, 'l1_rgb': l1_rgb, 'lpips': lpips, 'psnr': psnr}
 
        for k, v in log_dict.items():
            self.log(k, v, prog_bar=True)
 
        # Visualize results
        # img_pred = torch.cat([x['img_pred'] for x in outputs]).cpu()
        # img_real = torch.cat([x['img_real'] for x in outputs]).cpu()
        img_pred = outputs[0]['img_pred'].cpu()
        img_real = outputs[0]['img_real'].cpu()
 
        results = torch.cat(torch.cat([img_pred, img_real], dim=3).split(1, dim=0), dim=2)
        results_thumbnail = F.interpolate(results, scale_factor=0.9, mode='bilinear')[0]
 
        self.logger.log_image('results', [results_thumbnail], self.current_epoch)
 
    def configure_optimizers(self):
        opt = torch.optim.Adam(params=self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        # sch = torch.optim.lr_scheduler.StepLR(opt, step_size=35, gamma=0.1)
        return [opt], []
 
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
 
    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)