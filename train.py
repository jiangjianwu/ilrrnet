"""
Training Script of Indoor Context Understanding | 语义分割 + 布局合理度识别
Date: 2025.04.28
Author: fengbuxi@glut.edu.cn
"""
import os
import sys
import argparse
import time
import datetime
import shutil
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torchvision import transforms

from utils.distributed import get_rank, make_data_sampler, make_batch_data_sampler, synchronize
from utils.logger import setup_logger
from utils.lr_scheduler import WarmupPolyLR
from utils.loss import MixSoftmaxCrossEntropyLoss
from utils.score import SegmentationMetric

from dataloaders.dataloader import IndoorContextV2 as IndoorContext
from dataloaders.icr_classes import nclass
from models import get_model

def parse_args():
    parser = argparse.ArgumentParser(description='Indoor Context Understanding Training With Pytorch')
    parser.add_argument('--version', type=str, default='2.1', help='Model Version')
    parser.add_argument('--data_root', type=str, default='/mnt/dt01/datasets/indoor_rational_data_v2/', help='datasets root')
    parser.add_argument('--remark', type=str, default='rational_e3000', help='Remark')
    # training hyper params
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet18', 'resnet50',
                                 'resnet101', 'resnet152', 'densenet121',
                                 'densenet161', 'densenet169', 'densenet201'],
                        help='backbone name (default: vgg16)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 50)')
    parser.add_argument('--batch_size', type=int, default=6, metavar='N', help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--power', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-3, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--warmup-iters', type=int, default=0, help='warmup iters')
    parser.add_argument('--warmup-factor', type=float, default=1.0 / 3, help='lr = warmup_factor * lr')
    parser.add_argument('--warmup-method', type=str, default='linear', help='method of warmup')
    parser.add_argument('--base-size', type=int, default=520, help='base image size')
    parser.add_argument('--crop-size', type=int, default=480, help='crop image size')
    parser.add_argument('--workers', '-j', type=int, default=28, metavar='N', help='dataloader threads')
    parser.add_argument('--aux', action='store_true', default=False, help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,help='auxiliary loss weight')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Backbone Pretrained')
    parser.add_argument('--jpu', action='store_true', default=False, help='Backbone Pretrained')
    # cuda setting
    parser.add_argument('--gpus', type=str, default='0,1', help='GPU')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default='', help='put the path to resuming file if needed~/.torch/models/psp_resnet50_ade20k.pth')
    parser.add_argument('--save-epoch', type=int, default=10, help='save model every checkpoint-epoch')
    parser.add_argument('--log-iter', type=int, default=10, help='print log every log-iter')
    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=10, help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False, help='skip validation during training')
    parser.add_argument('--save_pred', action='store_true', default=True, help='save pred result during training')
    # parser.add_argument('--pths', type=str, default='2024-04-11_V2.0_rational_e3000', help='')
    parser.add_argument('--pths', type=str, default='2024-09-30_V2.1_rational_e3000', help='')

    
    args = parser.parse_args()

    # 设置训练采用的GPU情况
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        torch.backends.cudnn.enable =True
        torch.backends.cudnn.benchmark = True
        args.device = 'cuda:{}'.format(args.gpus.split(',')[0]) # 默认将第一个GPU作为主GPU
    else:
        args.device = 'cpu'

    return args

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.args.gpus = [int(i) for i in args.gpus.split(',')]
        self.version = args.version

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        train_dataset = IndoorContext(root=args.data_root, split='train', **data_kwargs)
        val_dataset = IndoorContext(root=args.data_root, split='val', **data_kwargs)
        args.iters_per_epoch = len(train_dataset) // (args.num_gpus * args.batch_size)
        args.max_iters = args.epochs * args.iters_per_epoch

        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iters)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                           pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        print(nclass)
        self.model = get_model(version=args.version, nclass=nclass, backbone=args.backbone, aux=args.aux, pretrained=args.pretrained, jpu=args.jpu, norm_layer=BatchNorm2d).to(self.device)
        # resume checkpoint if needed
        if args.resume != '':
            if os.path.exists(args.resume):
                _, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))
            else:
                print('Load checkpoint failed')
                sys.exit()

        # create criterion
        self.criterion_semseg = MixSoftmaxCrossEntropyLoss().to(self.device)
        self.criterion_rational = MixSoftmaxCrossEntropyLoss().to(self.device) # 二分类
        
        # optimizer, for model just includes pretrained, head and auxlayer
        params_list = list()
        if hasattr(self.model, 'pretrained'):
            params_list.append({'params': self.model.pretrained.parameters(), 'lr': args.lr})
        if hasattr(self.model, 'exclusive'):
            for module in self.model.exclusive:
                params_list.append({'params': getattr(self.model, module).parameters(), 'lr': args.lr * 10})
        self.optimizer = torch.optim.SGD(params_list,
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # lr scheduling
        self.lr_scheduler = WarmupPolyLR(self.optimizer,
                                         max_iters=args.max_iters,
                                         power=args.power,
                                         warmup_factor=args.warmup_factor,
                                         warmup_iters=args.warmup_iters,
                                         warmup_method=args.warmup_method)
        if self.args.device != 'cpu' and len(self.args.gpus) > 1: # 多GPU训练
            self.model = nn.DataParallel(self.model)
            cudnn.enable = True
            cudnn.benchmark = True

        # evaluation metrics
        self.metric_semseg = SegmentationMetric(len(train_dataset._elements))
        self.metric_rational = SegmentationMetric(len(train_dataset._layouts))

        self.best_pred = 0.0
        
    def train(self):
        save_to_disk = get_rank() == 0
        epochs, max_iters = self.args.epochs, self.args.max_iters
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.args.iters_per_epoch
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        start_time = time.time()
        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.model.train()

        for iteration, (images, semseg, rational, _) in enumerate(self.train_loader, self.args.start_epoch):
            iteration = iteration + 1
            self.lr_scheduler.step()

            images = images.to(self.device)
            semseg = semseg.to(self.device)
            rational = rational.to(self.device)

            outputs = self.model(images)

            # Semantic segmentation & Layout rational recognition Loss
            loss_dict = self.criterion_semseg(outputs[0:1], semseg)
            loss_semseg = sum(loss for loss in loss_dict.values())
            loss_dict = self.criterion_rational(outputs[1:2], rational)
            loss_rational = sum(loss for loss in loss_dict.values())
            
            # k1 = loss_rational / (loss_rational + loss_semseg)
            # k2 = loss_semseg / (loss_rational + loss_semseg)
            k1 = 1
            k2 = 1
            
            losses = k1 * loss_semseg + k2 * loss_rational

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                logger.info(
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || Loss_SemSeg: {:.4f}  || Loss_Rational: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                    iteration, max_iters, self.optimizer.param_groups[0]['lr'],
                    losses,
                    loss_semseg,
                    loss_rational,
                    str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))

            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.model, self.args, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation()
                self.model.train()

        save_checkpoint(self.model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info("Total training time: {} ({:.4f}s / it)".format(total_training_str, total_training_time / max_iters))

    def validation(self):
        is_best = False
        self.metric_semseg.reset()
        self.metric_rational.reset()
        
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        for i, (images, semseg, rational, _) in enumerate(self.val_loader):
            images = images.to(self.device)
            semseg = semseg.to(self.device)
            rational = rational.to(self.device)
            
            
            with torch.no_grad():
                outputs = model(images)
            
            # 精度验证
            self.metric_semseg.update(outputs[0], semseg)
            AllAcc5, ClsAcc5, IoU5, mIoU5 = self.metric_semseg.get()
            logger.info("Sample-SemSeg: {:d}, Validation AllAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, AllAcc5, mIoU5))
            logger.info("Sample-SemSeg: {:d}, ClassAcc: {}, IoU: {}".format(i + 1, ClsAcc5.__str__(), IoU5.__str__()))

            self.metric_rational.update(outputs[1], rational)
            AllAcc6, ClsAcc6, IoU6, mIoU6 = self.metric_rational.get()
            logger.info("Sample-Rational: {:d}, Validation AllAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, AllAcc6, mIoU6))
            logger.info("Sample-Rational: {:d}, ClassAcc: {}, IoU: {}".format(i + 1, ClsAcc6.__str__(), IoU6.__str__()))

        new_pred5 = (AllAcc5 + mIoU5) / 2
        new_pred6 = (AllAcc6 + mIoU6) / 2

        if new_pred5 > self.best_pred or new_pred6 > self.best_pred:
            is_best = True
            self.best_pred = max([new_pred5, new_pred6])
        save_checkpoint(self.model, self.args, is_best)
        synchronize()

def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.checkpoint)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}.pth'.format(args.backbone, args.aux)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = '{}_{}_best.pth'.format(args.backbone, args.aux)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)
 # Loss Optmization

if __name__ == '__main__':
    args = parse_args()

    # 生成运行目录
    save_dir = './running/'
    times = str(datetime.datetime.now().strftime('%Y-%m-%d'))
    if args.remark != '':
        timestr = times + "_V{}".format(args.version) + "_" + args.remark
    else:
        timestr = times + "_V{}".format(args.version)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    save_dir = Path(save_dir).joinpath('{}/'.format(timestr))
    save_dir.mkdir(exist_ok=True)
    log_dir = save_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    checkpoint_dir = save_dir.joinpath('checkpoint/')
    checkpoint_dir.mkdir(exist_ok=True)
    visual_dir = save_dir.joinpath('visual/')
    visual_dir.mkdir(exist_ok=True)
    args.checkpoint = checkpoint_dir.__str__()
    args.log_dir = log_dir
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        args.num_gpus = len(args.gpus.split(','))
    else:
       args.num_gpus = 1
    args.distributed = False
    args.lr = args.lr * args.num_gpus

    logger = setup_logger("indoor_context_understanding", args.log_dir, get_rank(), filename='train_{}_{}_log.txt'.format(args.backbone, args.aux))
    logger.info("Using {} GPUs".format(args.num_gpus))
    logger.info("Parameters:")
    logger.info(args)
    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
