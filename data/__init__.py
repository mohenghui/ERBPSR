import data.dataset as dataset
class Data(object):
    def __init__(self,cfg,transforms):
        super(Data, self).__init__()
        if cfg.train and not cfg.test:
            self.train_dataset=dataset.TrainDataset(cfg,transforms)
            self.val_dataset=dataset.ValidDataset(cfg,transforms)
            self.dataset=[self.train_dataset,self.val_dataset]
        elif cfg.eval and not cfg.test:
            self.dataset=dataset.TestDataset(cfg,transforms)
        elif cfg.test:
            self.dataset=dataset.TestDataset(cfg,transforms)
    def get_dataset(self):
        return self.dataset
    def __call__(self):
        return self.dataset