from scipy import misc
import torch
import datetime
import os
from utils.util import makedir
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import time
import numpy as np
from skimage import measure
import cv2
import math
import imageio
class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0
    def update(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg=self.sum/self.count

class timer():
    def __init__(self) :
        self.acc=0
        self.tic()
    def tic(self):
        self.t0=time.time() #开始时间
    def toc(self):
        return time.time()-self.t0 #结束时间
    def hold(self):
        self.acc+=self.toc()
    def release(self):
        ret=self.acc
        self.acc=0
        return ret
    def reset(self):
        self.acc=0

class checkpoint():
    def __init__(self,args):
        self.args=args
        self.ok=True
        self.log=torch.Tensor()
        now=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') #记录当前时间
        self.dir=os.path.join(args.ckp_dir,args.model,"x"+args.scale)
        makedir(self.dir)
        makedir(os.path.join(self.dir,"model"))
        makedir(os.path.join(self.dir,"val_results"))
        makedir(os.path.join(self.dir,"test_results"))
        open_type = 'a' if os.path.exists(os.path.join(self.dir + 'log.txt')) else 'w'
        self.log_file = open(os.path.join(self.dir , 'log.txt'), open_type)
        with open(os.path.join(self.dir , 'config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')
    def write_log(self,log,refresh=False):
        print(log)
        self.log_file.write(log+'\n')
        if refresh:
            self.log_file.close()
            self.log_file=open(os.path.join(self.dir,"log.txt"),"a")
    def done(self):
        self.log_file.close()
    def save_results(self,filename,save_list,scale,val_flag=True):
        if val_flag:
            filename=os.path.join(self.dir,"val_results",filename+"x"+scale)
        else:
            filename=os.path.join(self.dir,"test_results",filename+"x"+scale)
        normalized=save_list[0][0].data.mul(255/self.args.rgb_range)
        ndarr=normalized.byte().permute(1,2,0).cpu().numpy()
        imageio.imwrite('{}{}.png'.format(filename,'SR'),ndarr)
    def add_log(self, log):
        self.log = torch.cat([self.log, log])
def make_optimizer(args, my_model):
    trainable=filter(lambda x:x.requires_grad,my_model.parameters()) #需要反向传播的才进入优化器

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {
            'momentum': args.momentum,
            'weight_decay':0
        } #动量sgd
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'lr':args.adam_lr,
            'betas': (args.adam_beta1, args.adam_beta2),
            'eps': args.adam_epsilon,
            'weight_decay':0
        }
        # lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
        #          weight_decay=0,
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {
            'eps': args.epsilon,
            'weight_decay':0
            }
    elif args.optimizer=='ADAMW':
        optimizer_function=optim.AdamW
        # lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
        #          weight_decay=1e-2
        kwargs = {
            'lr':args.adamw_lr,
            'betas': (args.adamw_beta1, args.adamw_beta2),
            'eps': args.adamw_epsilon,
            'weight_decay':1e-2
        }

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    if args.decay_type=="step":
        scheduler=lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay_sr,
            gamma=args.gamma_sr,
        )
    elif args.use_multistep:
        milestones=list(map(lambda x:int (x),milestones))
        scheduler=lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma_sr
        )

    scheduler.step(args.start_epoch - 1)
    return scheduler
def quantize(img,rgb_range):
    pixel_range=255/rgb_range
    return img.mul(pixel_range).clamp(0,255).round().div(pixel_range)

def psnr(im1, im2):
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64)-min_val) / (max_val-min_val)
        return out
        
    im1 = im2double(im1)
    im2 = im2double(im2)
    psnr = measure.compare_psnr(im1, im2, data_range=1) #tcw 202105190332
    #psnr = measure.compare_psnr(im1, im2)
    return psnr

def calc_psnr(sr, hr, scale, rgb_range, benchmark=True):
    diff = (sr - hr).data.div(rgb_range)
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6
    import math
    shave = math.ceil(shave)
    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)
def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calc_ssim(img1, img2, scale=2, benchmark=False):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if benchmark:
        border = math.ceil(scale)
    else:
        border = math.ceil(scale) + 6

    img1 = img1.data.squeeze().float().clamp(0, 255).round().cpu().numpy()
    img1 = np.transpose(img1, (1, 2, 0))
    img2 = img2.data.squeeze().cpu().numpy()
    img2 = np.transpose(img2, (1, 2, 0))

    img1_y = np.dot(img1, [65.738, 129.057, 25.064]) / 255.0 + 16.0
    img2_y = np.dot(img2, [65.738, 129.057, 25.064]) / 255.0 + 16.0
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1_y = img1_y[border:h - border, border:w - border]
    img2_y = img2_y[border:h - border, border:w - border]

    if img1_y.ndim == 2:
        return ssim(img1_y, img2_y)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
