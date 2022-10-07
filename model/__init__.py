import torch.nn as nn
from importlib import import_module
import torch
import os
class Model(nn.Module):
    def __init__(self,args,ckp):
        super(Model, self).__init__()
        self.args = args
        self.cpu = args.cpu
        self.n_GPUs=args.n_GPUs
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        module = import_module('model.'+args.model)
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half': self.model.half()
        # self.load(
        #     ckp.dir,
        #     pre_train=args.pre_train,
        #     resume=args.resume,
        #     cpu=args.cpu
        # )
    def get_model(self):
        if self.n_GPUs <= 1 or self.cpu:
            return self.model
        else:
            return self.model.module
    def load(self, apath, pre_train='.', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume == -1:
            self.get_model().load_state_dict(
                torch.load(os.path.join(apath, 'model', 'model_latest.pt'), **kwargs),
                strict=True
            )

        elif resume == 0:
            if pre_train != '.':
                self.get_model().load_state_dict(
                    torch.load(pre_train, **kwargs),
                    strict=True
                )

        elif resume > 0:
            self.get_model().load_state_dict(
                torch.load(os.path.join(apath, 'model', 'model_{}.pt'.format(resume)), **kwargs),
                strict=False
            )
    def loss(self):
        pass