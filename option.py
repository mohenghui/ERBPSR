import yaml
import argparse
import os
from easydict import EasyDict
def main(config,args):
    pass
# if __name__ == '__main__':
parser = argparse.ArgumentParser(description='请输入超参数')
parser.add_argument('--model', default='HGSRCNN',help='choice your model')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--resume', type=bool, default=False,
                    help='resume from specific checkpoint')
parser.add_argument('--vis_dir', type=str, default=None, help='')
parser.add_argument('--rec', default='aster', choices=['aster', 'moran', 'crnn'])
parser.add_argument('--STN', action='store_true', default=False, help='')
parser.add_argument('--syn', action='store_true', default=False, help='use synthetic LR')
parser.add_argument('--mixed', action='store_true', default=False, help='mix synthetic with real LR')
parser.add_argument('--icdar2015', action='store_true', default=False, help='icdar2015 dataset')
parser.add_argument('--mask', action='store_true', default=False, help='')
parser.add_argument('--gradient', action='store_true', default=False, help='')
parser.add_argument('--laplace_gradient', action='store_true', default=False, help='')
parser.add_argument('--hd_u', type=int, default=32, help='')
parser.add_argument('--srb', type=int, default=5, help='')
parser.add_argument('--demo', action='store_true', default=False)
parser.add_argument('--demo_dir', type=str, default='./demo/demo5')
parser.add_argument("--patch_size", type=int, default=48)
parser.add_argument('--n_channels', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--hflip', action='store_true', default=True,
                    help='use horizontal flip')
parser.add_argument('--vflip', action='store_true', default=True,
                    help='use vertical flip')
parser.add_argument('--rot', action='store_true', default=True,
                    help='use rotate 90 degrees')
parser.add_argument('--train', action='store_true', default=False,
                    help='train sr model')
parser.add_argument('--eval', action='store_true', default=False,
                    help='eval sr model')
parser.add_argument('--test', action='store_true', default=True,
                    help='test sr model')
parser.add_argument('--train_data_dir', type=str, default="./datasets/DF2K",
                    help='test sr model')
parser.add_argument('--val_name', type=str, default="Set14")
parser.add_argument('--val_data_dir', type=str, default="./datasets/benchmark/Set14",
                    help='val sr model')
parser.add_argument('--test_data_dir', type=str, default="./testdata/Set14",
                    help='test sr model')
parser.add_argument('--test_name', type=str, default="Set14")
parser.add_argument('--scale', type=str, default='2',
                    help='super resolution scale')
parser.add_argument('--seed', type=str, default='1',
                    help='set random seed')
parser.add_argument("--num_gpu", type=int, default=1)
parser.add_argument("--multi_scale", type=bool, default=False)
parser.add_argument("--group", type=int, default=1)
parser.add_argument('--cpu', type=bool, default=False,
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--ckp_dir', type=str, default="./experiment",
                    help='experiment site')
parser.add_argument("--loss_fn", type=str, 
                        choices=["MSE", "L1", "SmoothL1"], default="1*L1")
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--optimizer', default='ADAMW',
                    choices=('SGD', 'ADAM', 'RMSprop','ADAMW'))
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--adam_beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--adam_beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument("--adam_lr", type=float, default=0.0001)

parser.add_argument('--adamw_beta1', type=float, default=0.9,
                    help='ADAMW beta1')
parser.add_argument('--adamw_beta2', type=float, default=0.999,
                    help='ADAMW beta2')
parser.add_argument('--adamw_epsilon', type=float, default=1e-8,
                    help='ADAMW epsilon for numerical stability')
parser.add_argument("--adamw_lr", type=float, default=0.0001)
parser.add_argument("--lr", type=float, default=0.0001)
# parser.add_argument('--test_only', action='store_true',default=
#                     help='set this option to test the model')
parser.add_argument('--lr_decay_sr', type=int, default=125,
                    help='learning rate decay per N epochs')
parser.add_argument("--decay_type",type=str,default='step',
                    choices=('step', 'half'),
                    help='choose your decay type')
parser.add_argument("--use_multistep",action='store_true', default=False)
parser.add_argument("--milestones",type=list,default=[100,150,200,300,500],
                    help='decay epoch')
parser.add_argument('--gamma_sr', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='resume from the snapshot, and the start_epoch')
parser.add_argument('--end_epoch', type=int, default=800,
                    help='resume from the snapshot, and the end_epoch')
parser.add_argument('--repeat', type=int, default=10,
                    help='repeat dataset')
parser.add_argument('--save_results', default=True,
                    help='save output results')
parser.add_argument('--pre_train', type=str, default= './experiment/HGSRCNN/x2/model/model_2.pt',
                    help='pre-trained model directory')
args = parser.parse_args()
config_path = os.path.join('config', 'super_resolution.yaml')
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)
# main(config, args)
