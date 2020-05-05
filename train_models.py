import os, sys, shutil, time, random, copy, json
import argparse

import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.utils.tensorboard import SummaryWriter
import lib.models as models
from lib.utils.utils import *

from lib.datasets.data_utils import get_data_statistics, generate_dataset
from lib.adversarial.adversarial import thermometer_encoding
import  art.defences as defences

import numpy as np

#get list of valid models from custom models directory
model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Train Models from Scratch', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='/nobackup/users/jzpan/datasets', help='path to dataset')
parser.add_argument('--dataset', type=str, help='choose dataset to benchmark adversarial training techniques on.')
parser.add_argument('--fold', type=int, help='if evaluating urbansound dataset, fold number to use for validation')
parser.add_argument('--arch', metavar='ARCH', default='resnet18', help='model architecture: to evaluate robustness on (default: resnet50)')
parser.add_argument('--thermometer', action='store_true', help='train model with thermometer encoding')
parser.add_argument('--level', type=int, default=10, help='level of thermometer encoding space to use')
parser.add_argument('--workers', type=int, default=16, help='number of data loading workers to use')
parser.add_argument('--pretrained', type=str, default='', help='path to pretrained model')
parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='comma-seperated string of gpu ids to use for acceleration (-1 for cpu only)')
parser.add_argument('--input_size', type=int, default=-1, help='input size of network; use -1 to use default input size')
parser.add_argument('--inc_contrast', type=float, default=1, help='factor to increase contrast for images')
# Hyperparameters
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer to use')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
parser.add_argument('--cosine', action='store_true', help='use cosine annealing schedule to decay learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--schedule', type=int, nargs='+', default=[83,123], help='list of epochs to reduce lr at')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='list of gammas to multiply with lr at each scheduled epoch; length of gammas should be the same as length of schedule')

# Model checkpoint flags
parser.add_argument('--print_freq', type=int, default=200, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='checkpoints', help='Folder to save checkpoints and log.')
parser.add_argument('--train', action='store_true', help='train the model')
parser.add_argument('--resume', type=str, default=None, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

global best_acc1, best_loss

# Below training loop, train function, and val function are adapted from: https://github.com/pytorch/examples/blob/master/imagenet/main.py

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if '-1' not in args.gpu_ids:
            inputs = inputs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        output = model(inputs)
        loss = criterion(output, target)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    summary.add_scalar('train acc5', top5.avg, epoch)
    summary.add_scalar('train acc1', top1.avg, epoch)
    summary.add_scalar('train loss', losses.avg, epoch)

def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix="Test: ")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (inputs, target) in enumerate(val_loader):
            if '-1' not in args.gpu_ids:
                inputs = inputs.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(inputs)
            print(inputs.shape)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f}'
             .format(top1=top1))

    summary.add_scalar('test acc5', top5.avg, epoch)
    summary.add_scalar('test acc1', top1.avg, epoch)
    summary.add_scalar('test loss', losses.avg, epoch)

    # visualize a batch of testing images
    dataiter = iter(val_loader)
    images, _ = dataiter.next()
    img_grid = utils.make_grid(images)
    summary.add_image("Validation Images", img_grid)

    return top1.avg, losses.avg

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    args = parser.parse_args()
    best_acc1=0
    best_loss=np.iinfo(np.int16).max
    
    # set gpus ids to use
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    #parse gpu ids from argument
    if torch.cuda.is_available():
        gpu_id_list = [int(i.strip()) for i in args.gpu_ids.split(',')]
    else:
        gpu_id_list = [-1]

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    # initialize tensorboard logger
    summary = SummaryWriter(args.save_path)
    print('==> Output path: {}...'.format(args.save_path))

    print(args)

    assert args.arch in model_names, 'Error: model {} not supported'.format(args.arch)

    # set variables based on dataset to evaluate on
    if args.dataset == 'imagenet':
        input_size = 224 if args.input_size == -1 else args.input_size
    elif args.dataset in ['cifar10', 'cifar100']: 
        input_size = 32 if args.input_size == -1 else args.input_size
    elif args.dataset in ['mnist', 'fmnist']:
        input_size = 28 if args.input_size == -1 else args.input_size 
    else:
        raise NotImplementedError

    criterion = torch.nn.CrossEntropyLoss()
    train_loader, test_loader, num_classes = generate_dataset(args.dataset, args.data_path, input_size, args.batch_size, args.workers, args.inc_contrast)
    
    if args.thermometer:
        with open('parameters/{}_parameters.json'.format(args.dataset)) as f:
            parameter_list = json.load(f)
        train_loader, test_loader = thermometer_encoding(train_loader, test_loader, parameter_list['thermometer'], save=True) 

    model = models.__dict__[args.arch](num_classes=num_classes, thermometer_encode=args.thermometer, level=args.level)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume) 
            # checkpointed models are wrapped in a nn.DataParallel, so rename the keys in the state_dict to match
            try:
                # our checkpoints save more than just the state_dict, but other checkpoints may only save the state dict, causing a KeyError
                checkpoint['state_dict'] = {n.replace('module.', '') : v for n, v in checkpoint['state_dict'].items()}
                model.load_state_dict(checkpoint['state_dict'])
            except KeyError:
                model.load_state_dict(checkpoint)
            print("=> loaded checkpoint '{}'" .format(args.resume))
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))
    else:
        print("=> Not using any checkpoint for {} model".format(args.arch))

    print(model)

    # wrap model in DataParallel for multi-gpu support
    if -1 not in gpu_id_list:
        model = torch.nn.DataParallel(model, device_ids = gpu_id_list)

    # define optimizer and set optimizer hyperparameters

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, 
                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5,factor=0.5,verbose=True)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate, momentum=args.momentum, 
                    weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
       optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                    weight_decay=args.weight_decay, nesterov=True)

    if -1 not in gpu_id_list and torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    cudnn.benchmark = True

    if args.evaluate:
        validate(test_loader, model, criterion, 1, args)

    if args.train:
        for epoch in range(args.epochs):

            # adjust lr using schedule for sgd
            if args.optimizer == 'sgd': 
                adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args)

            # evaluate on validation set
            acc1, val_loss = validate(test_loader, model, criterion, epoch, args)

            # adjust lr using ReduceLROnPlateau scheduler for adam
            if args.optimizer == 'adam':
                scheduler.step(np.around(val_loss,2))

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
        
            save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, args.save_path)

    summary.close()
