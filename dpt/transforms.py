import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target,landmarks):
        for t in self.transforms:
            image, target, landmarks = t(image, target, landmarks)
        
        return image, target, landmarks


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target, landmarks = None):
        size = random.randint(self.min_size, self.max_size)
        
        w,h = image.size
        
        image = F.resize(image, (size,size))
        target = F.resize(target, (size,size), interpolation=T.InterpolationMode.NEAREST)
        if landmarks is not None:
            new_w,new_h = image.size
            landmarks = landmarks * [np.round(new_w / w), np.round(new_h / h)]
            
        return image, target, landmarks


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target, landmarks = None):
        if random.random() < self.flip_prob:

            w,h = image.size
            image = F.hflip(image)
            target = F.hflip(target)
            if landmarks is not None:
                lm = landmarks.copy()
                lm[np.where(landmarks[:,1]<w/2),1] = landmarks[np.where(landmarks[:,1]<w/2),1]+w/2
                lm[np.where(landmarks[:,1]>w/2),1] = landmarks[np.where(landmarks[:,1]>w/2),1]-w/2
                landmarks = lm

        return image, target, landmarks
        


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, landmarks = None):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        if landmarks is not None:
            landmarks = landmarks-[crop_params[0],crop_params[1]]
            #landmarks = np.delete(landmarks,np.where(np.logical_or(np.logical_or(landmarks[:,0]<0,landmarks[:,1]<0),np.logical_or(landmarks[:,0]>=self.size[0],landmarks[:,1]>=self.size[1]))),0)
            
        return image, target, landmarks


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target,landmarks=None):
        h, w = image.shape[:2]
        
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        if landmarks is not None:
            landmarks = landmarks - [h/2-self.size[0]-h/2,w/2-self.size[1]]
            #landmarks = np.delete(landmarks,np.where(np.logical_or(np.logical_or(landmarks[:,0]<0,landmarks[:,1]<0),np.logical_or(landmarks[:,0]>=self.size[0],landmarks[:,1]>=self.size[1]))),0)
        return image, target, landmarks


class ToTensor:
    def __call__(self, image, target,landmarks=None):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target,landmarks


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target, landmarks=None):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target, landmarks


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target,landmarks = None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target, landmarks



class Grayscale:
    def __init__(self, num_channels):
        self.num_channels = num_channels


    def __call__(self, image, target,landmarks = None):
        image= F.rgb_to_grayscale(image, self.num_channels)
        return image, target, landmarks