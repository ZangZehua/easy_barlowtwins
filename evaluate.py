# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import time
import json
import urllib
import random
import signal
import argparse

from torch import nn, optim
from torchvision import models, datasets, transforms
import torch
import torchvision


def parse_args():

    parser = argparse.ArgumentParser(description='Evaluate resnet50 features on ImageNet')
    parser.add_argument('--data', type=str, metavar='DIR', default='/data/zzh/data',
                        help='path to dataset')
    parser.add_argument("--set", type=str, choices=['stl10', 'cifar10', 'cifar100', 'tiny'], default='stl10',
                        help='dataset')
    parser.add_argument('--pre', type=str, metavar='FILE', default=None,
                        help='path to pretrained model')
    parser.add_argument('--weights', type=str, default='freeze',
                        choices=('finetune', 'freeze'),
                        help='finetune or freeze resnet weights')
    parser.add_argument('--train_percent', type=int, default=100,
                        choices=(100, 10, 1),
                        help='size of traing set in percent')
    parser.add_argument('--workers', type=int, metavar='N', default=8,
                        help='number of data loader workers')
    parser.add_argument('--epochs', type=int, metavar='N', default=100,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int, metavar='N', default=256,
                        help='mini-batch size')
    parser.add_argument('--lr_backbone', type=float, metavar='LR', default=0.0,
                        help='backbone base learning rate')
    parser.add_argument('--lr_classifier', type=float, metavar='LR', default=0.3,
                        help='classifier base learning rate')
    parser.add_argument('--weight_decay', type=float, metavar='W', default=1e-6,
                        help='weight decay')
    parser.add_argument('--print_freq', type=int, metavar='N', default=100,
                        help='print frequency')
    parser.add_argument('--resume', type=str, default=None,
                        metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--model', type=str, default=None,
                        help='the model to test')
    parser.add_argument('--models', type=str, default="100, 1000, 100",
                        help='the models to test')
    parser.add_argument('--save_path', type=str, default="models_lp",
                        help='path to save model')
    parser.add_argument('--tb_path', type=str, default="runs_lp",
                        help='path to tensorboard')
    args = parser.parse_args()

    if args.set == "stl10":
        args.data = os.path.join(args.data, "STL-10")
    elif args.set == "cifar10":
        args.data = os.path.join(args.data, "CIFAR-10")
    elif args.set == "cifar100":
        args.data = os.path.join(args.data, "CIFAR-100")
    elif args.set == "tiny":
        args.data = os.path.join(args.data, "tiny-imagenet-200")
    elif args.set == "imagenet":
        args.data = os.path.join(args.data, "imagenet")
    else:
        raise FileNotFoundError

    save_path_base = os.path.join("saved", args.pre)
    args.save_path = os.path.join(save_path_base, args.save_path)
    args.tb_path = os.path.join(save_path_base, args.tb_path)
    args.pretrained = os.path.join(save_path_base, "models_pt/ckpt_" + args.model + ".pth")

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.isdir(args.tb_path):
        os.makedirs(args.tb_path)
    if not os.path.isdir(args.data_folder):
        raise ValueError('data path not exist: {}'.format(args.data_folder))

    return args


def main():
    args = parse_args()
    if args.train_percent in {1, 10}:
        args.train_files = urllib.request.urlopen(f'https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/{args.train_percent}percent.txt').readlines()
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
    # single-node distributed training
    args.rank = 0
    args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'
    args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = models.resnet50().cuda(gpu)
    state_dict = torch.load(args.pre, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    if args.weights == 'freeze':
        model.requires_grad_(False)
        model.fc.requires_grad_(True)
    classifier_parameters, model_parameters = [], []
    for name, param in model.named_parameters():
        if name in {'fc.weight', 'fc.bias'}:
            classifier_parameters.append(param)
        else:
            model_parameters.append(param)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    criterion = nn.CrossEntropyLoss().cuda(gpu)

    param_groups = [dict(params=classifier_parameters, lr=args.lr_classifier)]
    if args.weights == 'finetune':
        param_groups.append(dict(params=model_parameters, lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # automatically resume from checkpoint if it exists
    if args.resume.is_file():
        ckpt = torch.load(args.resume, map_location='cpu')
        start_epoch = ckpt['epoch']
        best_acc = ckpt['best_acc']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
    else:
        start_epoch = 0
        best_acc = argparse.Namespace(top1=0, top5=0)

    # Data loading code
    traindir = args.data / 'train'
    valdir = args.data / 'test'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.train_percent in {1, 10}:
        train_dataset.samples = []
        for fname in args.train_files:
            fname = fname.decode().strip()
            cls = fname.split('_')[0]
            train_dataset.samples.append(
                (traindir / cls / fname, train_dataset.class_to_idx[cls]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    kwargs = dict(batch_size=args.batch_size // args.world_size, num_workers=args.workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        # train
        if args.weights == 'finetune':
            model.train()
        elif args.weights == 'freeze':
            model.eval()
        else:
            assert False
        train_sampler.set_epoch(epoch)
        for step, (images, target) in enumerate(train_loader, start=epoch * len(train_loader)):
            output = model(images.cuda(gpu, non_blocking=True))
            loss = criterion(output, target.cuda(gpu, non_blocking=True))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % args.print_freq == 0:
                torch.distributed.reduce(loss.div_(args.world_size), 0)
                if args.rank == 0:
                    pg = optimizer.param_groups
                    lr_classifier = pg[0]['lr']
                    lr_backbone = pg[1]['lr'] if len(pg) == 2 else 0
                    stats = dict(epoch=epoch, step=step, lr_backbone=lr_backbone,
                                 lr_classifier=lr_classifier, loss=loss.item(),
                                 time=int(time.time() - start_time))

        # evaluate
        model.eval()
        if args.rank == 0:
            top1 = AverageMeter('Acc@1')
            top5 = AverageMeter('Acc@5')
            with torch.no_grad():
                for images, target in val_loader:
                    output = model(images.cuda(gpu, non_blocking=True))
                    acc1, acc5 = accuracy(output, target.cuda(gpu, non_blocking=True), topk=(1, 5))
                    top1.update(acc1[0].item(), images.size(0))
                    top5.update(acc5[0].item(), images.size(0))
            best_acc.top1 = max(best_acc.top1, top1.avg)
            best_acc.top5 = max(best_acc.top5, top5.avg)
            stats = dict(epoch=epoch, acc1=top1.avg, acc5=top5.avg, best_acc1=best_acc.top1, best_acc5=best_acc.top5)

        # sanity check
        if args.weights == 'freeze':
            reference_state_dict = torch.load(args.pretrained, map_location='cpu')
            model_state_dict = model.module.state_dict()
            for k in reference_state_dict:
                assert torch.equal(model_state_dict[k].cpu(), reference_state_dict[k]), k

        scheduler.step()
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1, best_acc=best_acc, model=model.state_dict(),
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
            torch.save(state, os.path.join(args.save_path, "ds_ckpt_" + args.epochs + ".pth"))


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
