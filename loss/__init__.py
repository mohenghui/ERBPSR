from bdb import effective
from importlib import import_module
from mimetypes import init
from turtle import forward
import torch
import torch.nn as nn
class Loss(nn.modules.loss._Loss):
    def __init__(self,args,ckp):
        super(Loss,self).__init__()
        self.n_GPUS=args.n_GPUs
        self.loss=[]
        self.loss_module=nn.ModuleList()
        for loss in args.loss_fn.split('+'):
            weight,loss_type=loss.split("*") #指的是loss所占权重
            if loss_type=="MSE":
                loss_function=nn.MSELoss()#均方差
            elif loss_type=="L1": 
                loss_function=nn.L1Loss()#L1loss
            elif loss_type=="CE":
                loss_function=nn.CrossEntropyLoss()#交叉熵
            elif loss_type.find('VGG')>=0:
                module=import_module('loss.vgg')
                loss_function=getattr(module,'VGG')(
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('GAN')>=0:
                module = import_module('loss.adversarial')
                loss_function=getattr(module,'Adversarial')(
                    args=args
                )
            self.loss.append({
                'type':loss_type,
                'weight':float(weight),
                'function':loss_function
            })
        for loss in self.loss:
            print("choose {} to be the loss".format(loss['type']))
            self.loss_module.append(loss['function'])
        self.log=torch.Tensor()

        device=torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half':self.loss_module.half()
    def forward(self,sr,hr):
        losses=[]
        for i, l in enumerate(self.loss):
            loss=l['function'](sr,hr)
            effective_loss=l['weight']*loss
            losses.append(effective_loss)
            self.log[-1,i]+=effective_loss.item()

        loss_sum=sum(losses)
        if len(self.loss)>1:
            self.log[-1,-1]+=loss_sum.item()
        return loss_sum
    def get_loss(self):
        if self.n_GPUS==1:
            return self.loss_module
        else:
            return self.loss_module.module
    def step(self):
        for l in self.get_loss():
            if hasattr(l,'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log=torch.cat((self.log,torch.zeros(1,len(self.loss))))
    
    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)
    
    def step(self):
        for l in self.get_loss():
            if hasattr(l,'scheduler'):
                l.scheduler.step()