import torch
import torch.nn as nn
import numpy as np
import random
import math
import warnings
import time
import os
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
#from augmentation.albumentation_dataset import Cifar10Albumentation, Cifar100Albumentation
#from augmentation.policies import get_baseline, get_auto_augmentation, get_rand_augmentation, \
#    get_album, test_transform, get_baseline_cutout, get_heavy_aug, get_equalize_aug, test_transform_equalize
from madgrad import MADGRAD
from timm.optim import RMSpropTF
from data.perovskite_dataset import PerovskiteDataset1d, PerovskiteDataset2d
from torchmetrics import F1, Precision, Recall, Accuracy, MeanAbsoluteError, MeanSquaredError

from data.augmentations import *


class BaseModel(pl.LightningModule):

    def __init__(self, hypparams):
        super(BaseModel, self).__init__()

        # Metrics
        '''self.train_f1 = F1(average='macro', num_classes=1, multiclass=False)
        self.train_precision = Precision(average='macro', num_classes=1, multiclass=False)
        self.train_recall = Recall(average='macro', num_classes=1, multiclass=False)
        self.train_acc = Accuracy()
        self.val_f1 = F1(average='macro', num_classes=1, multiclass=False)
        self.val_precision = Precision(average='macro', num_classes=1, multiclass=False)
        self.val_recall = Recall(average='macro', num_classes=1, multiclass=False)
        self.val_acc = Accuracy()'''

        self.train_MSE = MeanSquaredError()
        self.val_MSE = MeanSquaredError()
        self.train_MAE = MeanAbsoluteError()
        self.val_MAE = MeanAbsoluteError()

        # Training Args
        self.name = hypparams['name']
        self.batch_size = hypparams['batch_size']
        self.lr = hypparams['lr']
        self.weight_decay = hypparams['weight_decay']
        self.optimizer = hypparams['optimizer']
        self.nesterov = hypparams['nesterov']
        self.sam = hypparams['sam']
        self.adaptive_sam = hypparams['adaptive_sam']
        self.scheduler = hypparams['scheduler']
        self.T_max = hypparams['T_max']
        self.warmstart = hypparams['warmstart']
        self.epochs = hypparams['epochs']

        # Regularization techniques
        self.aug = hypparams['augmentation']
        self.mixup = hypparams['mixup']
        self.mixup_alpha = hypparams['mixup_alpha']  # 0.2
        self.label_smoothing = hypparams['label_smoothing']  # 0.1
        self.stochastic_depth = hypparams['stochastic_depth']  # 0.1 (with higher resolution maybe 0.2)
        self.resnet_dropout = hypparams['resnet_dropout']  # 0.5
        self.se = hypparams['squeeze_excitation']
        self.apply_shakedrop = hypparams['shakedrop']
        self.undecay_norm = hypparams['undecay_norm']
        self.zero_init_residual = hypparams['zero_init_residual']

        # Data and Dataloading
        self.data_dir = hypparams['data_dir']
        self.dataset = hypparams['dataset']
        self.num_workers = hypparams['num_workers']
        self.fold = hypparams['fold']
        self.dims = hypparams['dims']
        if self.dims == 1:
            self.train_mean, self.train_std = PerovskiteDataset1d(data_dir=self.data_dir, transform=None, fold=self.fold,
                                                                  split='train', label='PCE_mean').get_stats()
        elif self.dims == 2:
            self.train_mean, self.train_std = PerovskiteDataset2d(data_dir=self.data_dir, transform=None,
                                                                  fold=self.fold,
                                                                  split='train', label='PCE_mean').get_stats()

        os.makedirs(self.data_dir, exist_ok=True)
        self.download = False if any(os.scandir(self.data_dir)) else True

        # switch to manual optimization for Sharpness Aware Minimization
        if self.sam:
            self.automatic_optimization = False

        # Loss
        #self.criterion = nn.CrossEntropyLoss()#weight=class_weights)
        self.criterion = nn.MSELoss()

        # Inference
        self.softmax = nn.Softmax(dim=1)

        # Seed
        self.seed = hypparams['seed']

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)

        y_hat = y_hat.view(-1)

        loss = self.criterion(y_hat, y)

        '''# predict and save metrics
        with torch.no_grad():
            y_hat_norm = self.softmax(y_hat)
            if torch.isnan(y_hat_norm).any():
                print('######################################### Model predicts NaNs!')

        self.train_f1(y_hat_norm, y)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.train_precision(y_hat_norm, y)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.train_recall(y_hat_norm, y)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.train_acc(y_hat_norm, y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)'''

        self.train_MSE(y_hat, y)
        self.log('train_MSE', self.train_MSE, on_step=False, on_epoch=True, prog_bar=True)
        self.train_MAE(y_hat, y)
        self.log('train_MAE', self.train_MAE, on_step=False, on_epoch=True, prog_bar=True)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)

        # only if num_classes==1
        y_hat = y_hat.view(-1)

        val_loss = self.criterion(y_hat, y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)

        '''y_hat_norm = self.softmax(y_hat)

        self.val_f1(y_hat_norm, y)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.val_precision(y_hat_norm, y)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.val_recall(y_hat_norm, y)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.val_acc(y_hat_norm, y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)'''
        self.val_MSE(y_hat, y)
        self.log('val_MSE', self.val_MSE, on_step=False, on_epoch=True, prog_bar=True)
        self.val_MAE(y_hat, y)
        self.log('val_MAE', self.val_MAE, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_start(self):

        from models.resnet import BasicBlock, Bottleneck
        #from models.wide_resnet import BasicBlock as Wide_BasicBlock, Bottleneck as Wide_Bottleneck
        #from models.pyramidnet import BasicBlock as BasicBlock_pyramid, Bottleneck as Bottleneck_pyramid
        #from models.preact_resnet import PreActBlock, PreActBottleneck

        # TODO: disable weight init if model is pretrained once pretrained models are enabled
        print('Initializing weights')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        '''if self.zero_init_residual:

            if 'PreAct' in self.name:
                for m in self.modules():
                    if isinstance(m, PreActBottleneck):
                        nn.init.constant_(m.conv3.weight, 0)
                    elif isinstance(m, PreActBlock):
                        nn.init.constant_(m.conv2.weight, 0)

            elif 'ResNet' in self.name or 'WRN' in self.name:
                for m in self.modules():
                    if isinstance(m, Bottleneck) or isinstance(m, Wide_Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock) or isinstance(m, Wide_BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)

            elif 'Pyramid' in self.name:
                for m in self.modules():
                    if isinstance(m, Bottleneck_pyramid):
                        nn.init.constant_(m.bn4.weight, 0)
                    elif isinstance(m, BasicBlock_pyramid):
                        nn.init.constant_(m.bn3.weight, 0)'''

    def configure_optimizers(self):

        # leave bias and params of batch norm undecayed as in https://arxiv.org/pdf/1812.01187.pdf (Bag of tricks)
        if self.undecay_norm:
            model_params = []
            norm_params = []
            for name, p in self.named_parameters():
                if p.requires_grad:
                    if 'norm' in name or 'bias' in name or 'bn' in name:
                        norm_params += [p]
                    else:
                        model_params += [p]
            params = [{'params': model_params}, {'params': norm_params, 'weight_decay': 0}]
        else:
            params = self.parameters()

        if not self.sam:
            if self.optimizer=='SGD':
                optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9,
                                            weight_decay=self.weight_decay, nesterov=self.nesterov)
            elif self.optimizer=='Adam':
                optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
            elif self.optimizer=='AdamW':
                optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
            elif self.optimizer == 'Rmsprop':
                optimizer = RMSpropTF(params, lr=self.lr, weight_decay=self.weight_decay)
            elif self.optimizer == 'Madgrad':
                optimizer = MADGRAD(params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)

        else:
            pass
            '''# ASAM paper suggests 10x larger rho for adaptive SAM than in normal SAM
            rho = 0.5 if self.adaptive_sam else 0.05

            if self.optimizer=='SGD':
                base_optimizer = torch.optim.SGD
                optimizer = SAM(params, base_optimizer, adaptive=self.adaptive_sam, lr=self.lr, momentum=0.9,
                                weight_decay=self.weight_decay, nesterov=self.nesterov, rho=rho)
            elif self.optimizer=='Madgrad':
                base_optimizer = MADGRAD
                optimizer = SAM(params, base_optimizer, adaptive=self.adaptive_sam, lr=self.lr, momentum=0.9,
                                weight_decay=self.weight_decay, rho=rho)
            elif self.optimizer=='Adam':
                base_optimizer = torch.optim.Adam
                optimizer = SAM(params, base_optimizer, adaptive=self.adaptive_sam, lr=self.lr,
                                weight_decay=self.weight_decay, rho=rho)
            elif self.optimizer=='AdamW':
                base_optimizer = torch.optim.AdamW
                optimizer = SAM(params, base_optimizer, adaptive=self.adaptive_sam, lr=self.lr,
                                weight_decay=self.weight_decay, rho=rho)
            elif self.optimizer=='Rmsprop':
                base_optimizer = RMSpropTF
                optimizer = SAM(params, base_optimizer, adaptive=self.adaptive_sam, lr=self.lr,
                                weight_decay=self.weight_decay, rho=rho)'''

        if not self.scheduler:
            return [optimizer]
        else:
            if self.scheduler=='CosineAnneal' and self.warmstart==0:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max)
            elif self.scheduler == 'CosineAnneal' and self.warmstart > 0:
                scheduler = CosineAnnealingLR_Warmstart(optimizer, T_max=self.T_max, warmstart=self.warmstart)
            elif self.scheduler=='Step':
                # decays every 1/4 epochs
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.epochs // 4, gamma=0.1)
            elif self.scheduler=='MultiStep':
                # decays lr with 0.1 after half of epochs and 3/4 of epochs
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [self.epochs // 2, self.epochs * 3 // 4])

            return [optimizer], [scheduler]

    def train_dataloader(self):

        if self.aug == 'baseline':
            transform_train = baseline_2d(self.train_mean, self.train_std)

        elif self.aug == 'aug1':
            transform_train = aug1_2d(self.train_mean, self.train_std)

        '''elif self.aug == 'equalize':
            transform_train = get_equalize_aug(self.train_mean, self.train_std)

        elif self.aug == 'baseline_cutout':
            transform_train = get_baseline_cutout(cutout_size)

        elif self.aug == 'autoaugment':
            transform_train = get_auto_augmentation(cutout_size)

        elif self.aug == 'randaugment':
            transform_train = get_rand_augmentation(self.train_mean, self.train_std)

        elif self.aug == 'album':
            # Albumentation pipeline
            transform_train = get_album()'''

        if self.dataset == 'Perov_1d':

            trainset = PerovskiteDataset1d(data_dir=self.data_dir,
                                           transform=normalize_1d(self.train_mean, self.train_std), fold=self.fold,
                                           split='train', label='PCE_mean', val=False)

        elif self.dataset == 'Perov_2d':

            trainset = PerovskiteDataset2d(data_dir=self.data_dir,
                                           transform=transform_train, fold=self.fold,
                                           split='train', label='PCE_mean', val=False)

        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                                 pin_memory=True, worker_init_fn=seed_worker, persistent_workers=True)

        return trainloader

    def val_dataloader(self):

        if self.dataset == 'Perov_1d':
            valset = PerovskiteDataset1d(data_dir=self.data_dir,
                                         transform=normalize_1d(self.train_mean, self.train_std), fold=self.fold,
                                         split='train', label='PCE_mean', val=True)

        elif self.dataset == 'Perov_2d':
            valset = PerovskiteDataset2d(data_dir=self.data_dir,
                                         transform=normalize_2d(self.train_mean, self.train_std), fold=self.fold,
                                         split='train', label='PCE_mean', val=True)

        valloader = DataLoader(valset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                                pin_memory=True, worker_init_fn=seed_worker, persistent_workers=True)

        return valloader


class TimerCallback(Callback):

    def __init__(self, epochs, num_gpus):

        self.num_gpus = num_gpus
        if self.num_gpus > 0:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
        self.epochs = epochs
        self.epoch_times = []

    def on_train_epoch_start(self, trainer, pl_module):

        if trainer.current_epoch == 0:  # ignore first epoch
            pass
        else:
            if self.num_gpus == 0:
                self.start_cpu = time.time()
            else:
                self.start.record()

    def on_train_epoch_end(self, trainer, pl_module):#, outputs):

        if trainer.current_epoch == 0:
            pass
        else:
            if self.num_gpus == 0:
                end = time.time()
                elapsed_time = end - self.start_cpu
            else:
                self.end.record()
                torch.cuda.synchronize()
                elapsed_time = self.start.elapsed_time(self.end)/1000  # transform to seconds
            #print(elapsed_time)
            self.epoch_times.append(elapsed_time)
        if trainer.current_epoch==self.epochs-1:
            avg_epoch_time = np.mean(self.epoch_times)
            self.log('avg_epoch_time', avg_epoch_time)
            print('Average time per train epoch in seconds: ', avg_epoch_time)


def seed_worker(worker_id):
    '''
    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    to fix https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
    ensures different random numbers each batch with each worker every epoch while keeping reproducibility
    '''
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CosineAnnealingLR_Warmstart(_LRScheduler):
    '''
    Same as CosineAnnealingLR but includes a warmstart option that will gradually increase the LR
    for the amount of specified warmup epochs as described in https://arxiv.org/pdf/1706.02677.pdf
    '''

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False, warmstart=0):

        self.T_max = T_max - warmstart  # do not consider warmstart epochs for T_max
        self.eta_min = eta_min
        self.warmstart = warmstart
        self.T = 0

        super(CosineAnnealingLR_Warmstart, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        # Warmstart
        if self.last_epoch < self.warmstart:

            addrates = [(lr / (self.warmstart + 1)) for lr in self.base_lrs]
            updated_lr = [addrates[i] * (self.last_epoch + 1) for i, group in enumerate(self.optimizer.param_groups)]

            return updated_lr

        else:

            if self.T == 0:
                self.T += 1
                return self.base_lrs
            elif (self.T - 1 - self.T_max) % (2 * self.T_max) == 0:

                updated_lr = [group['lr'] + (base_lr - self.eta_min) *
                              (1 - math.cos(math.pi / self.T_max)) / 2
                              for base_lr, group in
                              zip(self.base_lrs, self.optimizer.param_groups)]

                self.T += 1
                return updated_lr

            updated_lr = [(1 + math.cos(math.pi * self.T / self.T_max)) /
                          (1 + math.cos(math.pi * (self.T - 1) / self.T_max)) *
                          (group['lr'] - self.eta_min) + self.eta_min
                          for group in self.optimizer.param_groups]

            self.T += 1
            return updated_lr


class ModelConstructor(BaseModel):
    def __init__(self, model, hypparams):
        super(ModelConstructor, self).__init__(hypparams)
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out

