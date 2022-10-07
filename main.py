import data
from option import args
import torch
from srwork import SRworker
import transforms 
import utility


if __name__=="__main__":
    torch.manual_seed(args.seed)
    checkpoint=utility.checkpoint(args)
    if checkpoint.ok:
        transform=transforms.SRTransforms(args)
        sr_dataset=data.Data(args,transform)
        my_dataset=sr_dataset.get_dataset()
        if not args.test and (args.n_GPUs==1 or args.cpu==True):
            trainer=SRworker(args,checkpoint,my_dataset,None)
            trainer.train()
        elif args.test and (args.n_GPUs==1 or args.cpu==True):
            tester=SRworker(args,checkpoint,None,my_dataset)
            tester.test()