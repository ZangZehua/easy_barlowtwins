import os
import math
import time
import random
import signal
import datetime
import argparse
import subprocess
import tensorboard

from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Barlow Twins Training')
    parser.add_argument('--data', type=str, metavar='DIR', default='/data/zzh/data',
                        help='path to dataset')
    parser.add_argument("--set", type=str, choices=['stl10', 'cifar10', 'cifar100', 'tiny'], default='stl10',
                        help='dataset')
    parser.add_argument('--workers', type=int, metavar='N', default=8,
                        help='number of data loader workers')
    parser.add_argument('--epochs', type=int, metavar='N', default=1000,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int, metavar='N', default=256,
                        help='mini-batch size')
    parser.add_argument('--learning_rate_weights', type=float, metavar='LR', default=0.2,
                        help='base learning rate for weights')
    parser.add_argument('--learning_rate_biases', type=float, metavar='LR', default=0.0048,
                        help='base learning rate for biases and batch norm parameters')
    parser.add_argument('--weight-decay', type=float, metavar='W', default=1e-6,
                        help='weight decay')
    parser.add_argument('--lambd', type=float, metavar='L', default=0.0051,
                        help='weight on off-diagonal terms')
    parser.add_argument('--projector', type=str, default='8192-8192-8192',
                        metavar='MLP', help='projector MLP')
    parser.add_argument('--print_freq', type=int, metavar='N', default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, metavar='N', default=10,
                        help='print frequency')
    parser.add_argument('--resume', type=str, default=None,
                        metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--model_path', type=str, default="models_pt",
                        help='path to save model')
    parser.add_argument('--tb_path', type=str, default="runs_pt",
                        help='path to tensorboard')
    args = parser.parse_args()

    pretrain_time = str(datetime.datetime.now().replace(microsecond=0).strftime("%Y%m%d-%H%M"))
    args.time = pretrain_time

    if args.set == "stl10":
        save_path_base = "saved/STL-10_" + pretrain_time
        args.data = os.path.join(args.data, "STL-10")
    elif args.set == "cifar10":
        save_path_base = "saved/CIFAR-10_" + pretrain_time
        args.data = os.path.join(args.data, "CIFAR-10")
    elif args.set == "cifar100":
        save_path_base = "saved/CIFAR-100_" + pretrain_time
        args.data = os.path.join(args.data, "CIFAR-100")
    elif args.set == "tiny":
        save_path_base = "saved/Tiny_" + pretrain_time
        args.data = os.path.join(args.data, "tiny-imagenet-200")
    elif args.set == "imagenet":
        save_path_base = "saved/IMAGENET_" + pretrain_time
        args.data = os.path.join(args.data, "imagenet")
    else:
        raise FileNotFoundError

    args.model_path = os.path.join(save_path_base, args.model_path)
    args.tb_path = os.path.join(save_path_base, args.tb_path)

    args.save_list = list(range(100, args.epochs, 100))
    args.save_list.extend(list(range(args.epochs - 90, args.epochs, 10)))

    if not os.path.isdir(save_path_base):
        os.makedirs(save_path_base)
        args.save_path_base = save_path_base
    if not os.path.isdir(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.isdir(args.tb_path):
        os.makedirs(args.tb_path)
    if not os.path.isdir(args.data_folder):
        raise ValueError('data path not exist: {}'.format(args.data_folder))

    return args


def main():  # no need for changing
    args = parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:58472'
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = 'tcp://localhost:58472'
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = BarlowTwins(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    # automatically resume from checkpoint if it exists
    if args.resume is not None and os.path.isfile(args.resume):
        print("===>loading resume", args.resume)
        ckpt = torch.load(args.resume, map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        print("===>train from 0")
        start_epoch = 0

    dataset = torchvision.datasets.ImageFolder(os.path.join(args.data, "unlabeled"), Transform())
    if args.rank == 0:
        print("===>n samples", len(dataset))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % args.print_freq == 0:
                if args.rank == 0:
                    # stats = dict(epoch=epoch, step=step,
                    #              lr_weights=optimizer.param_groups[0]['lr'],
                    #              lr_biases=optimizer.param_groups[1]['lr'],
                    #              loss=loss.item(),
                    #              time=int(time.time() - start_time))
                    print("epoch:{}, step:{}, lr_w:{}, lr_b:{}, loss:{}, time:{}".format(
                           epoch, step, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], loss.item(), int(time.time() - start_time)))
        if args.rank == 0 and epoch in args.save_list:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            save_path = os.path.join(args.save_path_base, "ckpt_" + str(epoch) + ".pth")
            print("===>saving", save_path)
            torch.save(state, save_path)
    if args.rank == 0:
        # save final model
        state = dict(epoch=args.epochs + 1, model=model.state_dict(),
                     optimizer=optimizer.state_dict())
        save_path = os.path.join(args.save_path_base, "ckpt_" + str(args.epoch) + ".pth")
        print("===>saving", save_path)
        torch.save(state, save_path)

        save_path = os.path.join(args.save_path_base, "ckpt_" + str(args.epoch) + "_final.pth")
        print("===>saving", save_path)
        torch.save(model.module.backbone.state_dict(), save_path)


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [2048] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss

    def embed(self, y1, y2, choice="backbone"):
        if choice == "backbone":
            z1 = self.backbone(y1)
            z2 = self.backbone(y2)
        elif choice == "projector":
            z1 = self.projector(self.backbone(y1))
            z2 = self.projector(self.backbone(y2))
        else:
            raise NotImplementedError
        return z1, z2


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])



class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


if __name__ == '__main__':
    main()
