"""
Eval Script
Date: 2023.11.04
Author: fengbuxi@glut.edu.cn
"""
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from utils.score import SegmentationMetric
from utils.logger import setup_logger
from utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler
from utils.visualize import get_color_pallete
from dataloaders.dataloader import IndoorContextV2 as IndoorContext
from dataloaders.icr_classes import nclass
from models import get_model
from train import parse_args
# nclass = nclass[4:6]
class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        val_dataset = IndoorContext(root=args.data_root, split='testval', **data_kwargs)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_model(version=args.version, nclass=nclass, backbone=args.backbone, aux=args.aux, pretrained=args.pretrained, jpu=args.jpu, norm_layer=BatchNorm2d).to(self.device)

        # if self.args.device != 'cpu' and len(self.args.gpus) > 1: # 多GPU训练
        #     self.model = nn.DataParallel(self.model)3
        self.model.load_state_dict(torch.load(args.checkpoints, map_location=args.device))
        
        # evaluation metrics
        self.metric_semseg = SegmentationMetric(len(val_dataset._elements))
        self.metric_rational = SegmentationMetric(len(val_dataset._layouts))

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def eval(self):
        # Reset mertric
        self.metric_semseg.reset()
        self.metric_rational.reset()

        self.model.eval()
        model = self.model
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        for i, (images, semseg, rational, filenames) in enumerate(self.val_loader):
            images = images.to(args.device)
            
            semseg = semseg.to(args.device)
            rational = rational.to(args.device)

            with torch.no_grad():
                outputs = model(images)
            
            # ================ M X N ==================
            self.metric_semseg.update(outputs[0], semseg)
            AllAcc5, ClsAcc5, IoU5, mIoU5 = self.metric_semseg.get()
            logger.info("Sample-SemSeg: {:d}, Validation AllAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, AllAcc5, mIoU5))
            logger.info("Sample-SemSeg: {:d}, ClassAcc: {}, IoU: {}".format(i + 1, ClsAcc5.__str__(), IoU5.__str__()))

            self.metric_rational.update(outputs[1], rational)
            AllAcc6, ClsAcc6, IoU6, mIoU6 = self.metric_rational.get()
            logger.info("Sample-Rational: {:d}, Validation AllAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, AllAcc6, mIoU6))
            logger.info("Sample-Rational: {:d}, ClassAcc: {}, IoU: {}".format(i + 1, ClsAcc6.__str__(), IoU6.__str__()))

            # Save pred results
            if self.args.save_pred:
                # Semseg task
                predict = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
                mask = get_color_pallete(predict, 'semseg')
                mask.save(os.path.join(args.visual, os.path.splitext(filenames[0])[0] + '_semseg.png'))
                # Rational task
                predict = torch.argmax(outputs[1], 1).squeeze(0).cpu().data.numpy()
                mask = get_color_pallete(predict, 'rational')
                mask.save(os.path.join(args.visual, os.path.splitext(filenames[0])[0] + '_rational.png'))


        synchronize()

if __name__ == '__main__':
    args = parse_args()
    args.distributed = False

    # Pretrained model path
    # args.save_dir = os.path.join('running', args.pths)
    args.save_dir = os.path.join('running', '2024-09-30_V2.1_rational_e3000')
    args.log_dir = os.path.join(args.save_dir, 'logs')
    args.checkpoints = os.path.join(args.save_dir, 'checkpoint', '{}_{}_best.pth'.format(args.backbone, args.aux))
    args.visual = os.path.join(args.save_dir, 'visual')
    logger = setup_logger("indoor_context_understanding", args.log_dir, get_rank(),
                          filename='{}_{}_eval_log.txt'.format(args.backbone, args.aux), mode='a+')

    evaluator = Evaluator(args)
    evaluator.eval()
    torch.cuda.empty_cache()