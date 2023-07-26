from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, SupConResNetProto
from losses import SupConLoss, SupConLossProto

import wandb

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass
for METHOD_i,EXP_NUM_i in zip(['SupCon', 'SupConProto'],[3,4]):
    EXP_NUM = EXP_NUM_i
    METHOD = METHOD_i  # 'SupCon' or 'SimCLR' or 'SupConProto'
    EPOCHS = 1000  # default 1000
    BATCH_SIZE = 256  # default 256
    MODEL = 'resnet50'  # default resnet50
    PROTO_AFTER_HEAD = True # default True
    DATASET = 'cifar100'
    EXP_NAME = f"EXP{EXP_NUM}: {METHOD}_{MODEL}_bs{BATCH_SIZE}_epochs{EPOCHS}{'' if PROTO_AFTER_HEAD else 'PROTO_BEFORE HEAD'}"  # f'exp{4} : SupConProto(v1)_bs256_epochs1000"

    HEAD = 'mlp'  # 'mlp' or 'linear'


    def parse_option():
        parser = argparse.ArgumentParser('argument for training')

        parser.add_argument('--print_freq', type=int, default=10,
                            help='print frequency')
        parser.add_argument('--save_freq', type=int, default=50,
                            help='save frequency')
        parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                            help='batch_size')
        parser.add_argument('--num_workers', type=int, default=8,
                            help='num of workers to use')
        parser.add_argument('--epochs', type=int, default=EPOCHS,
                            help='number of training epochs')

        # optimization
        parser.add_argument('--learning_rate', type=float, default=0.05,
                            help='learning rate')
        parser.add_argument('--lr_decay_epochs', type=str,
                            default=f'{int(7 / 10 * EPOCHS)},{int(8 / 10 * EPOCHS)},{int(9 / 10 * EPOCHS)}',
                            # '700,800,900'
                            help='where to decay lr, can be a list')
        parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                            help='decay rate for learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-4,
                            help='weight decay')
        parser.add_argument('--momentum', type=float, default=0.9,
                            help='momentum')

        # model dataset
        parser.add_argument('--model', type=str, default=MODEL)
        parser.add_argument('--dataset', type=str, default=DATASET,
                            choices=['cifar10', 'cifar100', 'path'], help='dataset')
        parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
        parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
        parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
        parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

        # method
        parser.add_argument('--method', type=str, default=METHOD,
                            choices=['SupCon', 'SupConProto', 'SimCLR'], help='choose method')

        # temperature
        parser.add_argument('--temp', type=float, default=0.07,
                            help='temperature for loss function')

        # other setting
        parser.add_argument('--cosine', action='store_true',
                            help='using cosine annealing')
        parser.add_argument('--syncBN', action='store_true',
                            help='using synchronized batch normalization')
        parser.add_argument('--warm', action='store_true',
                            help='warm-up for large batch training')
        parser.add_argument('--trial', type=str, default='0',
                            help='id for recording multiple runs')

        opt = parser.parse_args()
        opt.learning_rate = opt.learning_rate * opt.batch_size / 256
        # check if dataset is path that passed required arguments
        if opt.dataset == 'path':
            assert opt.data_folder is not None \
                   and opt.mean is not None \
                   and opt.std is not None

        # set the path according to the environment
        if opt.data_folder is None:
            opt.data_folder = './datasets/'
        opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
        opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

        iterations = opt.lr_decay_epochs.split(',')
        opt.lr_decay_epochs = list([])
        for it in iterations:
            opt.lr_decay_epochs.append(int(it))

        opt.model_name = f"{opt.method}_{opt.dataset}_{opt.model}_lr_{opt.learning_rate}_decay_{opt.weight_decay}_bsz_{opt.batch_size}_temp_{opt.temp}_trial_{opt.trial}" if EXP_NAME == '' else EXP_NAME

        if opt.cosine:
            opt.model_name = '{}_cosine'.format(opt.model_name)

        # warm-up for large-batch training,
        if opt.batch_size > 256:
            opt.warm = True
        if opt.warm:
            opt.model_name = '{}_warm'.format(opt.model_name)
            opt.warmup_from = 0.01
            opt.warm_epochs = 10
            if opt.cosine:
                eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
                opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                        1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
            else:
                opt.warmup_to = opt.learning_rate

        opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
        if not os.path.isdir(opt.tb_folder):
            os.makedirs(opt.tb_folder)

        opt.save_folder = os.path.join(opt.model_path, f'exp{EXP_NUM}')
        if not os.path.isdir(opt.save_folder):
            os.makedirs(opt.save_folder)

        if opt.dataset == 'cifar10':
            opt.n_cls = 10
        elif opt.dataset == 'cifar100':
            opt.n_cls = 100
        else:
            raise ValueError('dataset not supported: {}'.format(opt.dataset))

        return opt


    def set_loader(opt):
        # construct data loader
        if opt.dataset == 'cifar10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        elif opt.dataset == 'cifar100':
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)
        elif opt.dataset == 'path':
            mean = eval(opt.mean)
            std = eval(opt.std)
        else:
            raise ValueError('dataset not supported: {}'.format(opt.dataset))
        normalize = transforms.Normalize(mean=mean, std=std)

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        if opt.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                             transform=TwoCropTransform(train_transform),
                                             download=True)
        elif opt.dataset == 'cifar100':
            train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                              transform=TwoCropTransform(train_transform),
                                              download=True)
        elif opt.dataset == 'path':
            train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                                 transform=TwoCropTransform(train_transform))
        else:
            raise ValueError(opt.dataset)

        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

        return train_loader


    def set_model(opt):
        if opt.method in ['SupCon', 'SimCLR']:
            model = SupConResNet(name=opt.model, head=HEAD,n_cls=opt.n_cls)
            criterion = SupConLoss(temperature=opt.temp)
        elif opt.method in ['SupConProto']:
            model = SupConResNetProto(name=opt.model, head=HEAD,feat_dim=128,n_cls=opt.n_cls, proto_after_head= PROTO_AFTER_HEAD)
            criterion = SupConLossProto(temperature=opt.temp)

        # enable synchronized Batch Normalization
        if opt.syncBN:
            model = apex.parallel.convert_syncbn_model(model)

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model.encoder = torch.nn.DataParallel(model.encoder)
            model = model.cuda()
            criterion = criterion.cuda()
            cudnn.benchmark = True

        return model, criterion


    def train(train_loader, model, criterion, optimizer, epoch, opt):
        """one epoch training"""
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        for idx, (images, labels) in enumerate(train_loader):
            data_time.update(time.time() - end)

            images = torch.cat([images[0], images[1]], dim=0)
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # warm-up learning rate
            warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

            # compute loss
            if opt.method in ['SupCon', 'SimCLR']:

                features = model(images)
            elif opt.method in ['SupConProto']:
                features, proto_proj = model(images)

            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            if opt.method == 'SupCon':
                loss = criterion(features, labels)
            elif opt.method == 'SimCLR':
                loss = criterion(features)
            elif opt.method == 'SupConProto':
                # prototypes = model.prototypes

                # labels = torch.cat([labels, torch.arange(opt.n_cls).cuda()], dim=0)
                # old way to add prototypes (this creates too much prototypes)
                # prototypes = prototypes.unsqueeze(1)
                # prototypes = prototypes.repeat(1, 2, 1)
                # features = torch.cat([features, prototypes], dim=0)

                loss = criterion(features, labels, proto_proj)

            else:
                raise ValueError('contrastive method not supported: {}'.
                                 format(opt.method))

            # update metric
            losses.update(loss.item(), bsz)

            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (idx + 1) % opt.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))
                sys.stdout.flush()

        return losses.avg


    def main():
        opt = parse_option()

        print(opt.epochs)
        print(opt.batch_size)
        # build data loader
        train_loader = set_loader(opt)

        # build model and criterion
        model, criterion = set_model(opt)

        # build optimizer
        optimizer = set_optimizer(opt, model)

        # tensorboard
        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)


        # Initialize wandb:
        wandb.init(project=f"SupConPrototypes{opt.dataset}", name=opt.model_name, config=vars(opt))

        # training routine
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss = train(train_loader, model, criterion, optimizer, epoch, opt)
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            # tensorboard logger
            logger.log_value('loss', loss, epoch)
            logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            if epoch % opt.save_freq == 0:
                save_file = os.path.join(
                    opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save_model(model, optimizer, opt, epoch, save_file)
            wandb.log({"epoch": epoch, "Supcon/loss": loss, "Supcon/lr": optimizer.param_groups[0]['lr']})
        # save the last model
        save_file = os.path.join(
            opt.save_folder, 'last.pth')
        save_model(model, optimizer, opt, opt.epochs, save_file)
        wandb.finish()

if __name__ == '__main__':
    main()

