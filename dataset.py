import os
import numpy as np
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage import color


class VideoDataset(Dataset):
    def __init__(self, root, frame_stack, img_size=128, reference_size=None, 
                transform=None, target_transform=None, 
                normalize_lab=False, used_classes=None, intersect_frame_stacks=False,
                need_full_videos=False):
        self.root = root
        self.reference_size = reference_size                   # if reference_size is None, then not generate references
        self.frame_stack = frame_stack
        self.img_size = img_size
        self.loader = self.default_loader
        self.transform = transform
        self.target_transform = target_transform
        self.normalize_lab = normalize_lab                      # Flag for DeepRemaster-like normlization
        self.used_classes = used_classes                        # Indiced of classes that is used during construction of this dataset
        self.intersect_frame_stacks = intersect_frame_stacks    # Generate not distinct framestacks
        self.need_full_videos = need_full_videos                # len(dataset) = len(classes), each sample correspons to one video
        if self.need_full_videos:
            self.frame_stack = 1                

        classes, class_to_idx = self.find_classes(self.root)
        self.samples = self.make_dataset(root, class_to_idx, frame_stack)

    def process_image(self, path):
        img = self.loader(path).resize((self.img_size, self.img_size))
        # Convert to Lab color space
        image_lab = color.rgb2lab(img)
        # Convert to tensor and permute coordinates to format [C, H, W]
        image_l = torch.tensor(image_lab[:,:,0:1]).permute(2,0,1)
        image_ab = torch.tensor(image_lab[:,:,1:3]).permute(2,0,1)
        # Normalization like in DeepRemaster paper
        if self.normalize_lab:
            image_l = image_l / 100.                        #[0, 100] -> [0, 1]
            image_ab = (image_ab + 128).clip(0, 255) / 255. #[-127, 128] -> [0, 1]

        return image_l, image_ab

    def make_dataset(self, directory, class_to_idx, frame_stack):
        all_samples = []

        for target_class in tqdm(sorted(class_to_idx.keys())):
            class_index = class_to_idx[target_class]
            if self.used_classes is not None and class_index not in self.used_classes:
                # If class is not in the list of used classes - just skip it
                continue

            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue

            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                num_files = len(fnames)
                sorted_fnames = sorted(fnames)
                frame_idx = 0
                video_samples = []
                while num_files - frame_idx >= frame_stack:
                    frames = []
                    targets = []
                    for i in range(frame_stack):
                        path = os.path.join(root, sorted_fnames[frame_idx + i])
                        image_l, image_ab = self.process_image(path)
                        frames.append(image_l)
                        targets.append(image_ab)

                    sample = torch.stack(frames, dim=1)
                    target = torch.stack(targets, dim=1)
                    video_samples.append((sample, target))
                    if self.intersect_frame_stacks:
                        frame_idx += 1  # +=1 instead of += frame_stack to generate more diverse dataset
                    else:
                        frame_idx += frame_stack
                if self.need_full_videos:
                    # Divide back to wb-frabes and targets
                    frames = [x[0] for x in video_samples]
                    targets = [x[1] for x in video_samples]

                    sample = torch.cat(frames, dim=1)
                    target = torch.cat(targets, dim=1)

                    all_samples.append((sample, target))                    
                else:
                    # Just merge with all samples
                    all_samples += video_samples


        return all_samples

    def __getitem__(self, index):
        frames, targets = self.samples[index]
        if self.transform is not None:
            frames = self.transform(frames)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        if self.reference_size is None:
            return frames, targets

        reference_idx = np.random.randint(0, self.frame_stack, size=self.reference_size)
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
        # Use just pil_loader for simplification
        return self.pil_loader(path)

    def __len__(self):
        return len(self.samples)


class OneBWVideoDataset(Dataset):
    '''
        Dataset class for one large video
    '''
    def __init__(self, root, frame_stack, img_size=128,
                transform=None, target_transform=None, 
                normalize_lab=False):
        self.root = root
        self.frame_stack = frame_stack
        self.img_size = img_size
        self.loader = self.default_loader
        self.transform = transform
        self.target_transform = target_transform
        self.normalize_lab = normalize_lab                      # Flag for DeepRemaster-like normlization

        self.samples = self.make_dataset(root)

    def make_dataset(self, directory):
        samples = []

        img_paths = sorted(entry.name for entry in os.scandir(directory))
        if not img_paths:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        
        video_len = len(img_paths)
        frame_stack = self.frame_stack

        frame_idx = 0
        while video_len - frame_idx >= frame_stack:
            frames = []
            for i in range(frame_stack):
                path = os.path.join(directory, img_paths[frame_idx + i])
                img = self.loader(path).resize((self.img_size, self.img_size))
                # Convert to Lab color space
                image_lab = color.rgb2lab(img)
                # Convert to tensor and permute coordinates to format [C, H, W]
                image_l = torch.tensor(image_lab[:,:,0:1]).permute(2,0,1)
                # Normalization like in DeepRemaster paper
                if self.normalize_lab:
                    image_l = image_l / 100.                        #[0, 100] -> [0, 1]
                frames.append(image_l)

            sample = torch.stack(frames, dim=1)
            samples.append(sample)
            frame_idx += frame_stack
        
        return samples

    def __getitem__(self, index):
        frames = self.samples[index]
        if self.transform is not None:
            frames = self.transform(frames)

        return frames

    def pil_loader(self, path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def default_loader(self, path: str) -> Any:
        # Use just pil_loader for simplification
        return self.pil_loader(path)

    def __len__(self):
        return len(self.samples)