from fileinput import filename
import cv2
from torch.utils import data
import os
import imageio
import matplotlib.pyplot as plt
class TrainDataset(data.Dataset):
    def __init__(self,cfg,transforms):
        super(TrainDataset,self).__init__()
        self.cfg=cfg
        train_path=cfg.train_data_dir
        hr_path=os.path.join(train_path,"HR")
        lr_path=os.path.join(train_path,"LR_bicubic_X"+cfg.scale)
        self.hr=[]
        self.lr=[]
        self.filename=[]
        lr_list=os.listdir(lr_path)
        for file in os.listdir(hr_path):
            filename,tail=os.path.splitext(file)
            lr_filename=filename+"x"+cfg.scale+tail
            if lr_filename in lr_list:
                self.hr.append(os.path.join(hr_path,file))
                self.lr.append(os.path.join(lr_path,lr_filename))
                self.filename.append(filename)
        tmp_hr=self.hr[:]
        tmp_lr=self.lr[:]
        tmp_filename=self.filename[:]
        for _ in range(cfg.repeat-1):
            self.hr+=tmp_hr
        for _ in range(cfg.repeat-1):
            self.lr+=tmp_lr
        for _ in range(cfg.repeat-1):
            self.filename+=tmp_filename
        self.transform=transforms
    def __getitem__(self, index):
        hr_file_path=self.hr[index]
        hr_file_path=hr_file_path.replace("\\",'/')
        lr_file_path=self.lr[index]
        lr_file_path=lr_file_path.replace("\\",'/')
        filename=self.filename[index]
        hr = imageio.imread(hr_file_path)
        lr = imageio.imread(lr_file_path)
        if self.transform:
            lr,hr=self.transform.data_transform["train"](lr,hr)
        return lr,hr,filename
    def __len__(self):
        return len(self.hr)
class ValidDataset(data.Dataset):
    def __init__(self,cfg,transforms):
        super(ValidDataset,self).__init__()
        val_path=os.path.join(cfg.val_data_dir,"image_SRF_"+cfg.scale)
        self.hr=[]
        self.lr=[]
        self.filename=[]
        img_tail=".png"
        file_list=os.listdir(val_path)
        for file in file_list:
            tmp_sign="_SRF_"+cfg.scale+"_"
            filename,tail=file.split(tmp_sign)
            hr_filename=filename+tmp_sign+"HR"+img_tail
            lr_filename=filename+tmp_sign+"LR"+img_tail
            if lr_filename in file_list and hr_filename in file_list:
                self.hr.append(os.path.join(val_path,hr_filename))
                self.lr.append(os.path.join(val_path,lr_filename))
                self.filename.append(filename)
        self.transform=transforms
    def __getitem__(self, index):
        hr_file_path=self.hr[index]
        hr_file_path=hr_file_path.replace("\\",'/')
        lr_file_path=self.lr[index]
        lr_file_path=lr_file_path.replace("\\",'/')
        filename=self.filename[index]
        hr = imageio.imread(hr_file_path)
        lr = imageio.imread(lr_file_path)
        # hr = imageio.imread(hr_file_path).convert("RGB")
        # lr = imageio.imread(lr_file_path).convert("RGB")
        if self.transform:
            lr,hr=self.transform.data_transform["val"](lr,hr)
        return lr,hr,filename
    def __len__(self):
        return len(self.hr)
class TestDataset(data.Dataset):
    def __init__(self,cfg,transforms):
        super(TestDataset,self).__init__()
        lr_test_path=os.path.join(cfg.test_data_dir,"LRbicx"+cfg.scale)
        hr_test_path=os.path.join(cfg.test_data_dir,"original")
        self.hr=[]
        self.lr=[]
        self.filename=[]
        img_tail=".png"
        lr_file_list=os.listdir(lr_test_path)
        hr_file_list=os.listdir(hr_test_path)
        for file in hr_file_list:
            filename=os.path.basename(file)
            if filename in lr_file_list and filename in hr_file_list:
                self.hr.append(os.path.join(hr_test_path,filename))
                self.lr.append(os.path.join(lr_test_path,filename))
                self.filename.append(filename)
        self.transform=transforms
    def __getitem__(self, index):
        hr_file_path=self.hr[index]
        hr_file_path=hr_file_path.replace("\\",'/')
        lr_file_path=self.lr[index]
        lr_file_path=lr_file_path.replace("\\",'/')
        filename=self.filename[index]
        hr = imageio.imread(hr_file_path)
        lr = imageio.imread(lr_file_path)
        if self.transform:
            lr,hr=self.transform.data_transform["test"](lr,hr)
        return lr,hr,filename
    def __len__(self):
        return len(self.hr)