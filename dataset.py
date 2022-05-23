import os
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from dataset_utils import *


class VideoDataset(Dataset):
    def __init__(self, root, frame_stack, img_size=128, reference_size=5, extensions=None, transform=None,
                 target_transform=None):
        self.root = root
        self.reference_size = reference_size
        self.frame_stack = frame_stack
        self.img_size = img_size
        self.loader = self.default_loader
        self.extensions = extensions
        self.transform = transform
        self.target_transform = target_transform

        classes, class_to_idx = self.find_classes(self.root)
        self.samples = self.make_dataset(root, class_to_idx, frame_stack)

    def make_dataset(self, directory, class_to_idx, frame_stack):
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
                frame_idx = 0
                while num_files - frame_idx > frame_stack:
                    frames = []
                    targets = []
                    for i in range(frame_stack):
                        path = os.path.join(root, sorted_fnames[frame_idx + i])
                        img = self.loader(path).resize((self.img_size, self.img_size))
                        # img = Resize((self.img_size, self.img_size))(self.loader(path))
                        image_l, ab = convertRGB2LABTensor(img)
                        frames.append(image_l)
                        targets.append(ab)

                    sample = torch.stack(frames, dim=1)
                    target = torch.stack(targets, dim=1)
                    samples.append((sample, target))

                    frame_idx += frame_stack

        return samples

    def __getitem__(self, index):
        frames, targets = self.samples[index]
        if self.transform is not None:
            frames = self.transform(frames)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

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
        from torchvision import get_image_backend

        if get_image_backend() == "accimage":
            print("!!!!")
            #return accimage_loader(path)
        else:
            return self.pil_loader(path)

    def __len__(self):
        return len(self.samples)
