import transforms.transforms as transforms
class SRTransforms(object):
    def __init__(self,args):
        super(SRTransforms, self).__init__()
        self.data_transform = {
        "train": transforms.Compose([
            transforms.SetChannel(args.n_channels),
            transforms.GetPatch(args.patch_size,int(args.scale)),
            transforms.Augment(args.hflip,args.vflip,args.rot),
            transforms.ToTensor(),

        ]),
        "val": transforms.Compose([
            transforms.SetChannel(args.n_channels),
            transforms.ToTensor(),
        ]),
        "test": transforms.Compose([
            transforms.SetChannel(args.n_channels),
            transforms.ToTensor(),
        ])
        }