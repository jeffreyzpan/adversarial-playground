import os, sys, shutil, time, random, copy
import argparse

import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as utils
from tensorboardX import SummaryWriter
import lib.models as models
from lib.utils.utils import *
from lib.utils.stock_dataset import generate_stocks_dataset
from lib.utils.adversarial import gen_attacks, gen_defences

from art.classifiers import PyTorchClassifier
import art.attacks.evasion as evasion
import art.defences as defences
import numpy as np

#get list of valid models from custom models directory
model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Adversarial Training Benchmarking', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='datasets/', help='path to dataset')
parser.add_argument('--dataset', type=str, choices=['stocks', 'xrays', 'gtsrb', 'esc'], help='choose dataset to benchmark adversarial training techniques on.')
parser.add_argument('--stock', type=str, help='if evaluating stocks dataset, stock to evaluate on')
parser.add_argument('--window_size', type=int, default=10, help='number of time steps to predict in the future for stocks')
parser.add_argument('--arch', metavar='ARCH', default='resnet50', help='model architecture: to evaluate robustness on (default: resnet50)')
parser.add_argument('--workers', type=int, default=16, help='number of data loading workers to use')
parser.add_argument('--pretrained', type=str, default='', help='path to pretrained model')
parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='comma-seperated string of gpu ids to use for acceleration (-1 for cpu only)')
# Hyperparameters
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
parser.add_argument('--cosine', action='store_true', help='use cosine annealing schedule to decay learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')

# Model checkpoint flags
parser.add_argument('--print_freq', type=int, default=200, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='checkpoints', help='Folder to save checkpoints and log.')
parser.add_argument('--train', action='store_true', help='train the model')
parser.add_argument('--resume', type=str, default=None, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

# Experiments
parser.add_argument('--eval_attacks', action='store_true', help='evaluate attacks on model')
parser.add_argument('--attacks', type=str, default='fgsm,pgd,hopskipjump,deepfool', help='comma seperated string of attacks to evaluate')
parser.add_argument('--defences', type=str, default='jpeg', help='comma seperated string of defences to evaluate')

global best_acc1

# Below training loop, train function, and val function are adapted from: https://github.com/pytorch/examples/blob/master/imagenet/main.py

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    if args.dataset == 'stocks':
        mse = AverageMeter('MSE', ':.4e')
        #progress = 
    top1 = AverageMeter('Acc@1', ':6.2f')
    if args.dataset != 'xrays':
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))
    else:
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1],
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
        if args.dataset == 'xrays':
            acc1, _ = accuracy(output, target, topk=(1, 1))
        else:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        if args.dataset != 'xrays':
            top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    summary.add_scalar('train acc1', top1.avg, epoch)
    #summary.add_scalar('train acc5', top5.avg, epoch)
    summary.add_scalar('train loss', losses.avg, epoch)


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    if args.dataset != 'xrays':
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix="Test: ")
    else:
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1],
            prefix="Test: ")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if '-1' not in args.gpu_ids:
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            if args.dataset == 'xrays':
                acc1, _ = accuracy(output, target, topk=(1, 1))
            else:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            if args.dataset != 'xrays':
                top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))
        summary.add_scalar('test acc1', top1.avg, epoch)
        #summary.add_scalar('test acc5', top5.avg, epoch)
        summary.add_scalar('test loss', losses.avg, epoch)

    return top1.avg

if __name__ == '__main__':
    args = parser.parse_args()
    best_acc1=0
    
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

    print(model_names)
    assert args.arch in model_names, 'Error: model {} not supported'.format(args.arch)

    # set variables based on dataset to evaluate on
    if args.dataset == 'xrays':
        num_classes = 2
        # apply resizing and other transforms to dataset
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.Resize((224, 224)), transforms.ToTensor()])
        test_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()])
        input_shape = (1, 224, 224)
        criterion = torch.nn.CrossEntropyLoss()

        # create dataset
        trainset = dset.ImageFolder(os.path.join(args.data_path, args.dataset, 'train'), transform=train_transform)
        testset = dset.ImageFolder(os.path.join(args.data_path, args.dataset, 'val'), transform=test_transform) 

    if args.dataset == 'gtsrb':
        num_classes = 43
        # apply resizing and normalize to mean=0, std=1 (adapted from https://github.com/poojahira/gtsrb-pytorch)
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))])
        test_transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()])
        input_shape = (3, 32, 32)
        criterion = torch.nn.CrossEntropyLoss()

        # create dataset
        trainset = dset.ImageFolder(os.path.join(args.data_path, args.dataset, 'train'), transform=train_transform)
        testset = dset.ImageFolder(os.path.join(args.data_path, args.dataset, 'val'), transform=test_transform) 

    if args.dataset == 'stocks':
        trainset, testset = generate_stocks_dataset(os.path.join(args.data_path, args.dataset), args.stock, args.window_size)
        optimizer = torch.nn.MSELoss()

    # initialize dataloaders

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    if args.dataset == 'stocks':
        model = models.__dict__[args.arch](window_size=args.window_size)
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

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

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                    weight_decay=args.weight_decay, nesterov=True)

    if -1 not in gpu_id_list and torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    cudnn.benchmark = True

    # perform attacks and defences on dataset
    if args.eval_attacks:
        attack_name_list = args.attacks.split(',')
        attack_name_list = [i.strip().lower() for i in attack_name_list] #sanitize inputs

        defence_name_list = args.defences.split(',')
        defence_name_list = [i.strip().lower() for i in defence_name_list] #sanitize inputs

        attack_list = {}
        defence_list = {}

        # initialize attacks and append to dict

        classifier = PyTorchClassifier(model=copy.deepcopy(model), clip_values=(0,1), loss=criterion, optimizer=optimizer, input_shape=input_shape, nb_classes=num_classes) 

        if 'fgsm' in attack_name_list:
            attack_list['fgsm'] = evasion.FastGradientMethod(classifier, targeted=False, eps=0.05)
        if 'pgd' in attack_name_list:
            attack_list['pgd'] = evasion.ProjectedGradientDescent(classifier, targeted=False, max_iter=10, eps_step=0.1, eps=0.3)
        if 'hopskipjump' in attack_name_list:
            attack_list['hsj'] = evasion.HopSkipJump(classifier)
        if 'query-efficient' in attack_name_list:
            raise NotImplementedError
        if 'deepfool' in attack_name_list:
            attack_list['deepfool'] = evasion.DeepFool(classifier)

        # initialize defenses and append to dict

        if 'thermometer' in defence_name_list:
            defence_list['thermometer'] = defences.ThermometerEncoding(clip_values=(0,1)) 
        if 'pixeldefend' in defence_name_list:
            defence_list['pixeldefend'] = defences.PixelDefend(clip_values=(0,1)) 
        if 'tvm' in defence_name_list:
            defence_list['tvm'] = defences.TotalVarMin(clip_values=(0,1))
        if 'saddlepoint' in defence_name_list:
            defence_list['saddlepoint'] = defences.AdversarialTrainer(classifier, attacks=attack_list['pgd'])
        if 'jpeg' in defence_name_list:
            defence_list['jpeg'] = defences.JpegCompression(clip_values=(0,1), channel_index=1)

        # get initial validation set accuracy

        validate(test_loader, model, criterion, 1, args)

        # ART appears to only support numpy arrays, so convert dataloader into a numpy array of images
        image_batches, label_batches = zip(*[batch for batch in test_loader])
        test_images = torch.cat(image_batches).numpy()
        test_labels = torch.cat(label_batches).numpy()

        adv_dict = gen_attacks(test_images, test_labels, classifier, criterion, attack_list)

        # loop through all generated dataloaders with adversarial images
        for attack_name in adv_dict:

            #measure attack success
            print("Testing performance of attack {}: ".format(attack_name))
            validate(adv_dict[attack_name], model, criterion, 1, args)

            adv_images, _ = zip(*[batch for batch in adv_dict[attack_name]])
            adv_images = torch.cat(adv_images).numpy()

            # save adv images for visualization purposes
            #import pdb
            #pdb.set_trace()
            save_images(adv_images, args.save_path, 'attack_{}'.format(attack_name))

            adv_images = adv_images/255.

            print("Generating defences for attack {}: ".format(attack_name))

            def_adv_dict = gen_defences(test_images, adv_images, attack_name, test_labels, classifier, criterion, defence_list)

            for def_name in def_adv_dict:
                print("Testing performance of defence {}: ".format(def_name))
                validate(def_adv_dict[def_name], model, criterion, 1, args)
                import pdb
                pdb.set_trace()
                def_images , _ = zip(*[batch for batch in def_adv_dict[def_name]])
                def_images = torch.cat(def_images).numpy()

                # save def images for visualization purposes
                save_images(def_images, args.save_path, '{}_{}'.format(attack_name, def_name))

    if args.evaluate:
        validate(test_loader, model, criterion, 1, args)

    if args.train:
        for epoch in range(args.epochs):

            adjust_learning_rate(optimizer, epoch, args)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args)

            # evaluate on validation set
            acc1 = validate(test_loader, model, criterion, epoch, args)

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
