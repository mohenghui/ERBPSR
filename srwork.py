from decimal import Decimal
from tkinter.tix import Tree
from torch.utils.data import _utils
from torch.utils.data import DataLoader
import tqdm
import model
import os
import torch
import utility
import loss
class SRworker():
    def __init__(self,args,ckp,tv_dataset,test_dataset):
        super(SRworker, self).__init__()
        self.train_val_dataset=tv_dataset
        self.test_dataset=test_dataset
        self.args=args
        self.ckp=ckp
        self.model=model.Model(args, ckp).get_model()
        self.loss= loss.Loss(args, ckp) if self.args.train else None
        self.optimizer=utility.make_optimizer(args,self.model)#定义优化器
        self.scheduler=utility.make_scheduler(args,self.optimizer)#学习率调度算法
        self.nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
        if args.resume:
            self.resume()
    def train(self):
        batch_size=self.args.batch_size
        # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        train_dataloader=DataLoader(self.train_val_dataset[0],
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=True,
                        num_workers=self.nw,
                        collate_fn=_utils.collate.default_collate)

        if self.args.resume:
            pass
        end_epoch=self.args.end_epoch
        while self.scheduler.last_epoch+1<=end_epoch:
            self.scheduler.step()
            self.loss.step()
            epoch=self.scheduler.last_epoch+1
            lr=self.args.lr*(self.args.gamma_sr**((epoch)// self.args.lr_decay_sr))
            for param_group in self.optimizer.param_groups:
                param_group['lr']=lr
            self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch,Decimal(lr)))
            self.loss.start_log()
            self.model.train()
            timer=utility.timer()
            losses_contrast,losses_sr=utility.AverageMeter(),utility.AverageMeter()
            for batch,(lr,hr,_) in enumerate(train_dataloader):
                lr=lr.cuda()
                hr=hr.cuda()
                self.optimizer.zero_grad()
                timer.tic()#开始计时
                sr=self.model(lr)
                loss_sr=self.loss(sr,hr)
                losses_sr.update(loss_sr.item())
                loss_sr.backward()
                self.optimizer.step()
                timer.hold()#记录一次时间的累计
                self.ckp.write_log(
                    'Epoch: [{:04d}][{:04d}/{:04d}]\t'
                    'Loss [SR loss:{:.3f}]\t'
                    'Time [{:.1f}s]'.format(
                        epoch,(batch+1)*self.args.batch_size,len(self.train_val_dataset[0]),
                        losses_sr.avg,
                        timer.release(),
                    )   
                )
            self.val(epoch)
            self.loss.end_log(len(train_dataloader))

            #save model
            # target=self.model.get_model()
            # model_dict=target.state_dict()
            save_files={
                'model':self.model.state_dict(),
                'optimizer':self.optimizer.state_dict(),
                'lr_scheduler':self.scheduler.state_dict(),
                'epoch':epoch
            }
            torch.save(save_files,os.path.join(self.ckp.dir,'model','model_{}.pt'.format(epoch)))
    def val(self,epoch):
        scale=int(self.args.scale)
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 1))
        val_dataloader=DataLoader(self.train_val_dataset[1],
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=self.nw,
        collate_fn=_utils.collate.default_collate)
        self.model.eval()
        timer_test = utility.timer()
        with torch.no_grad():
            eval_psnr = 0
            eval_ssim = 0
            for batch,(lr,hr,filename) in enumerate(val_dataloader):
                lr=lr.cuda()
                hr=hr.cuda()
                hr = self.crop_border(hr, scale)
                timer_test.tic()
                sr=self.model(lr)
                timer_test.hold()
                sr=utility.quantize(sr,self.args.rgb_range)
                hr=utility.quantize(hr,self.args.rgb_range)

                #cal psnr and ssim
                eval_psnr+=utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range,
                        benchmark=Tree
                    )
                eval_ssim += utility.calc_ssim(
                    sr, hr, scale,
                    benchmark=True
                )
                if self.args.save_results:
                    save_list=[sr]
                    filename=filename[0]
                    self.ckp.save_results(filename,save_list,str(scale))
            self.ckp.log[-1,0]=eval_psnr/len(val_dataloader)
            self.ckp.write_log(
                '[Epoch {}---{} x{}]\tPSNR: {:.3f} SSIM:{:.4f}'.format(
                    epoch,
                    self.args.val_name,
                    str(scale),
                    eval_psnr/len(val_dataloader),
                    eval_ssim/len(val_dataloader)
                )
            )
                # val_loss_sr=self.loss(sr,hr)
                # self.ckp.write_log("val loss [{:04d}]".format(val_loss_sr))
    def resume(self):
        checkpoint=torch.load(self.args.pre_train,map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch=checkpoint['epoch']+1
        print("The model start epoch is [{}]".format(start_epoch-1))

    def test(self):
        scale=int(self.args.scale)
        self.resume()
        test_dataloader=DataLoader(self.test_dataset,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
                num_workers=self.nw,
                collate_fn=_utils.collate.default_collate)
        timer_test = utility.timer()
        with torch.no_grad():
            eval_psnr = 0
            eval_ssim = 0
            for batch,(lr,hr,filename) in enumerate(test_dataloader):
                lr=lr.cuda()
                hr=hr.cuda()
                hr = self.crop_border(hr, scale)
                timer_test.tic()
                sr=self.model(lr)
                timer_test.hold()
                sr=utility.quantize(sr,self.args.rgb_range)
                hr=utility.quantize(hr,self.args.rgb_range)

                #cal psnr and ssim
                eval_psnr+=utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range,
                        benchmark=Tree
                    )
                eval_ssim += utility.calc_ssim(
                    sr, hr, scale,
                    benchmark=True
                )
                if self.args.save_results:
                    save_list=[sr]
                    filename=filename[0]
                    self.ckp.save_results(filename,save_list,str(scale),False)
            print("Use [{}] test [{}x{}] \t PSNR: {:.3f} SSIM:{:.4f}".format(
                os.path.basename(self.args.pre_train),self.args.test_name,
                    str(scale),
                    eval_psnr/len(test_dataloader),
                    eval_ssim/len(test_dataloader)))
    def crop_border(self, img_hr, scale):
        # if type(scale)==
        b, c, h, w = img_hr.size()

        img_hr = img_hr[:, :, :int(h//scale*scale), :int(w//scale*scale)]

        return img_hr