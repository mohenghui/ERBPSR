import numpy as np
import skimage.color as sc
from data.common import set_channel,get_patch,augment,np2Tensor
class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, lr, hr):
        for t in self.transforms:
            lr, hr = t(lr, hr)
        return lr, hr

class SetChannel(object):
    def __init__(self, n_channels=3):
        self.n_channels=n_channels
    def __call__(self, lr, hr):
        lr=set_channel(lr,self.n_channels)
        hr=set_channel(hr,self.n_channels)
        return lr,hr

class GetPatch(object):
    def __init__(self, patch_size=48, scale=1):
        self.patch_size=patch_size
        self.scale=scale
    def __call__(self, lr, hr):
        lr,hr=get_patch(lr,hr,self.patch_size,self.scale)
        return lr,hr

class Augment(object):
    def __init__(self,hflip=True,vflip=True,rot=True):
        self.hflip=hflip
        self.vflip=vflip
        self.rot=rot
    def __call__(self, lr, hr):
        lr=augment(lr,self.hflip,self.vflip,self.rot)
        hr=augment(hr,self.hflip,self.vflip,self.rot)
        return lr,hr
        
class ToTensor(object):
    def __call__(self, lr, hr):
        lr = np2Tensor(lr)
        hr = np2Tensor(hr)
        return lr, hr