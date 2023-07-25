from typing import Any
import torch
import dpt.transforms as T
from torchvision.transforms import functional as F
from torchvision import transforms as Tr

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2 * base_size)

        transforms = []
        transforms.append(T.RandomResize(min_size, max_size))
        if hflip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(hflip_prob))
        transforms.extend(
            [
                T.RandomCrop(crop_size),
                T.ToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )

        self.transforms = T.Compose(transforms)

    def __call__(self, image, anno, landmarks=None):
        return self.transforms(image, anno, landmarks)
    

class SegmentationPresetVal:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.base_size = base_size
        transforms = []
        transforms.extend(
            [
                T.ToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )

        self.transforms = T.Compose(transforms)

    def __call__(self, image, anno, landmarks=None):
        image = F.resize(image, (self.base_size, self.base_size))
        anno = F.resize(anno, (self.base_size, self.base_size), interpolation=Tr.InterpolationMode.NEAREST)
        return self.transforms(image, anno, landmarks)
