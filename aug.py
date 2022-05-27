import albumentations as A
import torch.nn.functional as F
import random
import numpy as np
import torchvision.transforms as transforms

transform_reference_train = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.5, 0.8)),
    transforms.RandomHorizontalFlip()
])

transform_reference_test = transforms.Compose([
    transforms.Resize(128)
])

def transform_train(left_images, right_images):
    crop_size = 128
    width, height = transforms.functional.get_image_size(left_images)
    random_top = np.random.randint(0, height - crop_size + 1)
    random_left = np.random.randint(0, width - crop_size + 1)

    left_images = transforms.functional.crop(left_images, random_top, random_left, crop_size, crop_size)
    right_images = transforms.functional.crop(right_images, random_top, random_left, crop_size, crop_size)
    
    p = random.random()
    if p < 0.5:
        left_images = transforms.functional.hflip(left_images)
        right_images = transforms.functional.hflip(right_images)
    
    random_rotate = np.random.uniform(-5, 5)
    left_images = transforms.functional.rotate(left_images, random_rotate)
    right_images = transforms.functional.rotate(right_images, random_rotate)
        
    return left_images, right_images

def transform_test(left_images, right_images):
    resize_size = 128

    left_images = transforms.functional.resize(left_images, resize_size)
    right_images = transforms.functional.resize(right_images, resize_size)
        
    return left_images, right_images