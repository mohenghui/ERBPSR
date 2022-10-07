import random
import numpy as np
import skimage.color as sc
import torch

import matplotlib.pyplot as plt
def set_channel(img, n_channels=3):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    c = img.shape[2]
    if n_channels == 1 and c == 3:
        img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
    elif n_channels == 3 and c == 1:
        img = np.concatenate([img] * n_channels, 2)

    return img

# def get_patch(img, patch_size=48, scale=1):
#     th, tw = img.shape[:2]  ## HR image

#     tp = round(scale * patch_size)

#     tx = random.randrange(0, (tw-tp))
#     ty = random.randrange(0, (th-tp))

#     return img[ty:ty + tp, tx:tx + tp, :]

def get_patch(lr, hr, size, scale=1):
    h, w = lr.shape[:2] #h,w,channel [:-1] beside the final element, such as channel 
    x = random.randint(0, w-size-1) #random number
    y = random.randint(0, h-size-1)

    hsize = size*scale
    hx, hy = x*scale, y*scale

    crop_lr = lr[y:y+size, x:x+size].copy() #low-resolution patch
    crop_hr = hr[hy:hy+hsize, hx:hx+hsize].copy()#high-resolution patch
    return crop_lr, crop_hr

def np2Tensor(img, rgb_range=255):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)

    return tensor


def augment(img, hflip=True,vflip=True, rot=True):
    hflip = hflip and random.random() < 0.5 #水平翻转
    vflip = vflip and random.random() < 0.5 #垂直翻转
    rot90 = rot and random.random() < 0.5 #旋转90

    if hflip: img = img[:, ::-1, :]
    if vflip: img = img[::-1, :, :]
    if rot90: img = img.transpose(1, 0, 2)

    return img
