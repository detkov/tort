import random

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms
from torchvision.transforms.functional import rotate


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class TortAugmenter(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, masked_crop_scale=None):
        self.apply_masking = masked_crop_scale is not None

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            # transforms.ToTensor(),
            normalize,
        ])
        # second global crop
        self.global_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            # transforms.ToTensor(),
            normalize,
        ])
        # masked crop
        if self.apply_masking:
            self.masked_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                normalize,
                # transforms.ToTensor(),
                transforms.RandomErasing(1, masked_crop_scale, (0.75, 1))
            ])

        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            # transforms.ToTensor(),
            normalize,
        ])

    def __call__(self, image):
        crops = [self.global_transform1(image), 
                 self.global_transform2(image)]
        if self.apply_masking:
            crops.append(self.masked_transform(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops


def rand_rot(batch, rot_prob):
    if np.random.random() > rot_prob:
        return batch, np.zeros(len(batch))

    rots = np.random.randint(1, 4, size=len(batch))
    for idx in range(len(batch)):
        batch[idx] = rotate(batch[idx], int(90 * rots[idx]))
    
    return batch, rots
