from __future__ import print_function

import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn

from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.resnet_big import SupConResNet, LinearClassifier,SupConResNetProto

import torch.nn as nn

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

import wandb
# Rappel : ici les prototypes ne servent qu'à initialiser les poids du classifieurs, si  PREDICT_WITH_PROTO = True,
# on utilise simplement les prototypes comme poids initiaux du classifieur, et on freeze le backbone
METHOD = 'SupConProto'  # 'SupCon' or 'SimCLR' or 'SupConProto'
PROTO_AFTER_HEAD = True # has to be true if the pretrained model is a SupConProto model with proto_after_head=True
DATASET = 'cifar100'  # default cifar10
MODEL = 'resnet50'  # default resnet18

PRETRAINING_EPOCHS = 200
EXP_NUMBER = 3
EXP_NAME = f'exp{EXP_NUMBER} LE: {PRETRAINING_EPOCHS} epochs'
PREDICT_WITH_PROTO = False #if True, simply init the FC weights with proto, if not random init, not real interest since the aim of prototypes is mostly to init the FC weights
NO_GRAD = False # if True, freeze the classifier (backbone is always frozen) pour evaluer la classif en produit scalaire avec les protos direct sans les rentrainer
BS = 128  # default 128 ou 256
EPOCHS = 100  # default 100
CKPT = f'./save/SupCon/{DATASET}_models/exp{EXP_NUMBER}/ckpt_epoch_{PRETRAINING_EPOCHS}.pth' # default 'last.pth' or 'ckpt_epoch_100.pth'


if PREDICT_WITH_PROTO :
    EXP_NAME += '_predWithProto'
if NO_GRAD :
    EXP_NAME += '_noGrad'
    EPOCHS = 10
if not PREDICT_WITH_PROTO:
    PROTO_AFTER_HEAD = False
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=BS,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default=MODEL)
    parser.add_argument('--dataset', type=str, default=DATASET,
                        choices=['cifar10', 'cifar100'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default=CKPT,
                        help='path to pre-trained model')

    opt = parser.parse_args()
    opt.method = METHOD
    # set the path according to the environment
    opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'. \
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size) if EXP_NAME == '' else EXP_NAME

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
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

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_model(opt):
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']
    #print(state_dict.n_cls)
    if opt.method in ['SupCon', 'SimCLR']:
        model = SupConResNet(name=opt.model,n_cls=opt.n_cls)
    elif opt.method in ['SupConProto']:
        model = SupConResNetProto(name=opt.model, feat_dim=128, n_cls=opt.n_cls,
                                  proto_after_head=PROTO_AFTER_HEAD)
    else:
        raise ValueError('contrastive method not supported: {}'.
                         format(opt.method))

    criterion = torch.nn.CrossEntropyLoss()
    if PREDICT_WITH_PROTO:
        classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls,prototypes = model.prototypes)
    else:
        classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
            if PROTO_AFTER_HEAD: # when prototypes are after head, features have to live in contrastive space
                features = model.head(features)

        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        if not NO_GRAD:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info

        wandb.log({
            "Train/CELoss": losses.avg,
            "Train/Acc@1": top1.avg,
        })
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            features = model.encoder(images)
            if PROTO_AFTER_HEAD: # when prototypes are after head, features have to live in contrastive space
                features = model.head(features)
            output = classifier(features)

            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # Initialize wandb:
    wandb.init(project=f"SupConPrototypes{DATASET}", name=f"{opt.model_name}", config=vars(opt))

    # training routine

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()

        with torch.set_grad_enabled(not NO_GRAD):
            train_loss, train_acc = train(train_loader, model, classifier, criterion,
                                      optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, train_acc))

        # log training metrics

        # eval for one epoch
        with torch.set_grad_enabled(not NO_GRAD):
            val_loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc

        # log validation metrics
        wandb.log({"epoch": epoch, "Train/CELoss": train_loss, "Train/Acc": train_acc, "Val/CELoss": val_loss,
                   "Val/Acc": val_acc, "Val/BestValAcc": best_acc})

        print('best accuracy: {:.2f}'.format(best_acc))



if __name__ == '__main__':
    main()
