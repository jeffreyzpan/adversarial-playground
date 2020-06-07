import os
import sys
import shutil
import time
import random
import copy
import json
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
from lib.adversarial.adversarial import *
from lib.adversarial.tvm import TotalVarMin

from art.classifiers import PyTorchClassifier
import art.defences as defences
import numpy as np
import foolbox as fb
import eagerpy as ep

# get list of valid models from custom models directory
model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Adversarial Training Benchmarking',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str,
                    default='/nobackup/users/jzpan/datasets', help='path to dataset')
parser.add_argument('--dataset', type=str,
                    help='choose dataset to benchmark adversarial training techniques on.')
parser.add_argument('--arch', metavar='ARCH', default='resnet50',
                    help='model architecture: to evaluate robustness on (default: resnet50)')
parser.add_argument('--workers', type=int, default=16,
                    help='number of data loading workers to use')
parser.add_argument('--pretrained', type=str, default='',
                    help='path to pretrained model')
parser.add_argument('--gpu_ids', type=str, default='0,1,2,3',
                    help='list of gpu ids to use for acceleration (-1 for cpu only)')
# Hyperparameter for Adversarial Retrainings
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to perform adversarial training for')
parser.add_argument('--optimizer', type=str,
                    default='sgd', help='optimizer to use')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.1, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=0.0001, help='weight decay')
parser.add_argument('--schedule', type=int, nargs='+',
                    default=[83, 123], help='list of epochs to reduce lr at')
parser.add_argument('--gammas', type=float, nargs='+', default=[
                    0.1, 0.1], help='list of gammas to multiply with lr at each scheduled epoch; length of gammas should be the same as length of schedule')
parser.add_argument('--pretrained_adv', action='store_true', help='whether to use pretrained adv_trained model')

# Model checkpoint flags
parser.add_argument('--print_freq', type=int, default=200,
                    metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='checkpoints',
                    help='Folder to save checkpoints and log.')
parser.add_argument('--resume', type=str, default=None, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', type=int, default=0,
                    metavar='N', help='manual epoch number (useful on restarts)')

# Experiments
parser.add_argument('--attacks', type=str, nargs='+', default=[
                    'fgsm', 'pgd', 'deepfool', 'bim'], help='list of attacks to evaluate')
parser.add_argument('--epsilons', type=float, nargs='+', default=[2/255, 4/255, 8/255, 16/255], help='epsilon values to use for attacks')
parser.add_argument('--defences', type=str, nargs='+', default=[], help='list of defences to evaluate')
parser.add_argument('--input_size', type=int, default=-1,
                    help='input size for adv training; use -1 to use default input size')
parser.add_argument('--inc_contrast', type=float, default=1, help='factor to increase the dataset contrast')

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
                inputs = inputs.to(f'cuda:{args.gpu_ids}', non_blocking=True)
                target = target.to(f'cuda:{args.gpu_ids}', non_blocking=True)

            # compute output
            output = model(inputs)
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
    assert len(gammas) == len(
        schedule), "length of gammas and schedule should be equal"
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
    best_acc1 = 0
    best_loss = np.iinfo(np.int16).max

    # set gpus ids to use
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # check that CUDA is actually available and pass in GPU ids, use CPU if not
    if torch.cuda.is_available():
        gpu_id_list = [int(i.strip()) for i in args.gpu_ids.split(',')]
    else:
        gpu_id_list = [-1]

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.isdir(os.path.join(args.save_path, 'adv_trained_model')):
        os.makedirs(os.path.join(args.save_path, 'adv_trained_model'))

    # initialize tensorboard logger
    summary = SummaryWriter(args.save_path)
    print('==> Output path: {}...'.format(args.save_path))

    print(args)

    assert args.arch in model_names, 'Error: model {} not supported'.format(
        args.arch)

    # set variables based on dataset to evaluate on
    if args.dataset == 'imagenet':
        input_size = 224 if args.input_size == -1 else args.input_size
    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
        input_size = 32 if args.input_size == -1 else args.input_size
    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        input_size = 28 if args.input_size == -1 else args.input_size

    criterion = torch.nn.CrossEntropyLoss()
    train_loader, test_loader, num_classes = generate_dataset(
        args.dataset, args.data_path, input_size, args.batch_size, args.workers, args.inc_contrast)

    mean, std = get_data_statistics(train_loader)

    model = models.__dict__[args.arch](num_classes=num_classes)
    assert os.path.isfile(
        args.resume), 'Adversarial benchmarking requires a pretrained model — use train_models.py to train a model'
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    # checkpointed models are wrapped in a nn.DataParallel, so rename the keys in the state_dict to match
    try:
        # our checkpoints save more than just the state_dict, but other checkpoints may only save the state dict, causing a KeyError
        checkpoint['state_dict'] = {
             n.replace('module.', ''): v for n, v in checkpoint['state_dict'].items()}
        model.load_state_dict(checkpoint['state_dict'])
    except KeyError:
        model.load_state_dict(checkpoint)
    print("=> loaded checkpoint '{}'" .format(args.resume))
    print(model)

    # wrap model in DataParallel for multi-gpu support
    if -1 not in gpu_id_list:
        model = torch.nn.DataParallel(model, device_ids=gpu_id_list)

    # define optimizer and set optimizer hyperparameters

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate,
                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=5, factor=0.5, verbose=True)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate, momentum=args.momentum,
                    weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
       optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                    weight_decay=args.weight_decay, nesterov=True)

    if -1 not in gpu_id_list:
        model.to(f'cuda:{args.gpu_ids}')
        criterion.to(f'cuda:{args.gpu_ids}')

    cudnn.benchmark = True
    if args.dataset in ['fmnist', 'mnist']:
        input_shape = (1, input_size, input_size)
    else:
        input_shape = (3, input_size, input_size)

    # get initial validation set accuracy

    initial_acc, _ = validate(test_loader, model, criterion, 1, args)

    # perform attacks and defences on dataset

    attack_list = {}
    defence_list = {}

    # initialize attacks and append to dict

    classifier = fb.PyTorchModel(copy.deepcopy(model).eval(), (0, 1))

    with open('parameters/{}_parameters.json'.format(args.dataset)) as f:
        parameter_list = json.load(f)
    epsilons = args.epsilons

    #white box attacks
    if 'fgsm' in args.attacks:
        attack_list['fgsm'] = fb.attacks.FGSM()
    if 'carliniLinf' in args.attacks:
        # Use ART implementation as Foolbox doesn't have Linf CW attack
        art_classifier = PyTorchClassifier(copy.deepcopy(model), loss=criterion, optimizer=optimizer, input_shape=input_shape, nb_classes=num_classes)
        cw_dict = cw_linf(art_classifier, test_loader, epsilons)
        
    if 'pgd' in args.attacks:
        attack_list['pgd'] = fb.attacks.PGD()
    if 'deepfool' in args.attacks:
        attack_list['deepfool'] = fb.attacks.LinfDeepFoolAttack()
    if 'bim' in args.attacks:
        attack_list['bim'] = fb.attacks.LinfBasicIterativeAttack()

    #black box attacks

    if 'boundary' in args.attacks:
        attack_list['boundary'] = fb.attacks.BoundaryAttack()
    if 'saltandpepper' in args.attacks:
        attack_list['saltandpepper'] = fb.attacks.SaltAndPepperNoiseAttack()
    if 'gaussian' in args.attacks:
        attack_list['gaussian'] = fb.attacks.L2RepeatedAdditiveGaussianNoiseAttack()
    if 'uniform' in args.attacks:
        attack_list['uniform'] = fb.attacks.L2RepeatedAdditiveUniformNoiseAttack()

    # initialize defences and append to dict

    if 'pixeldefend' in args.defences:
        
        pixel_cnn = models.__dict__['pixel_cnn'](input_channels=input_shape[0])
        assert os.path.isfile('./pixel_models/{}/model_best.pth.tar'.format(args.dataset)), 'PixelDefend requires a pretrained PixelCNN++, please add the checkpoint in the appropriate folder'
        print("=> loading checkpoint './pixel_models/{}/model_best.pth.tar'".format(args.dataset))

        checkpoint = torch.load('./pixel_models/{}/model_best.pth.tar'.format(args.dataset))
        checkpoint = {n.replace('module.', ''): v for n, v in checkpoint.items()}
        pixel_cnn.load_state_dict(checkpoint)
        print("loaded PixelCNN++")

        if -1 not in gpu_id_list:
            pixel_cnn = torch.nn.DataParallel(pixel_cnn, device_ids=gpu_id_list).cuda()
        pixel_cnn = PyTorchClassifier(pixel_cnn, loss=criterion, optimizer=optimizer, input_shape=input_shape, nb_classes=num_classes)

        pixel_params = parameter_list['pixeldefend']
        defence_list['pixeldefend'] = defences.PixelDefend(clip_values=(
            pixel_params['clip_min'], pixel_params['clip_max']), eps=pixel_params['eps'], pixel_cnn=pixel_cnn)

    if 'tvm' in args.defences:
        tvm_params = parameter_list['tvm']
        defence_list['tvm'] = TotalVarMin(clip_values=(
            tvm_params['clip_min'], tvm_params['clip_max']), prob=tvm_params['prob'], lamb=tvm_params['lamb'], max_iter=tvm_params['max_iter'])
    if 'jpeg' in args.defences:
        jpeg_params = parameter_list['jpeg']
        defence_list['jpeg'] = defences.JpegCompression(clip_values=(
            jpeg_params['clip_min'], jpeg_params['clip_max']), channel_index=jpeg_params['channel_index'], quality=jpeg_params['quality'])
    if 'adv_retraining' in args.defences: 
        print("performing adversarial retraining using Madry's method")
        if args.pretrained_adv:
            robust_model = copy.deepcopy(model)
            assert os.path.isfile(os.path.join(args.save_path, 'adv_trained_model/checkpoint.pth.tar')), 'no adversarially-trained model found!'
            print('=> loading adversarially-trained model')

            checkpoint = torch.load(os.path.join(args.save_path, 'adv_trained_model/checkpoint.pth.tar'))
            model.load_state_dict(checkpoint['state_dict']) 
        else:
            pgd_attack = {'madry_pgd': fb.attacks.PGD()}
            adv_train_loader = gen_attacks(train_loader, classifier, pgd_attack, [8/255])['madry_pgd'][0]
            adv_test_loader = gen_attacks(test_loader, classifier, pgd_attack, [8/255])['madry_pgd'][0]
            robust_model = copy.deepcopy(model)

            # define optimizer and set optimizer hyperparameters

            if args.optimizer == 'adam':
                robust_optimizer = torch.optim.Adam(robust_model.parameters(), args.learning_rate,
                        weight_decay=args.weight_decay)
                robust_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        robust_optimizer, 'min', patience=5, factor=0.5, verbose=True)
            elif args.optimizer == 'rmsprop':
                robust_optimizer = torch.optim.RMSprop(robust_model.parameters(), args.learning_rate, momentum=args.momentum,
                            weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                robust_optimizer = torch.optim.SGD(robust_model.parameters(), args.learning_rate, momentum=args.momentum,
                            weight_decay=args.weight_decay, nesterov=True)

            top1 = 0
            for epoch in range(args.epochs):
                train(adv_train_loader, robust_model,
                      criterion, robust_optimizer, epoch, args)
                acc1, val_loss = validate(
                      adv_test_loader, robust_model, criterion, epoch, args)
                if args.optimizer == 'adam':
                    robust_scheduler.step(np.around(val_loss, 2))

                # remember best acc@1 and save checkpoint
                top1 = max(top1, acc1)
                is_best = acc1 > top1
                save_checkpoint({
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': robust_model.state_dict(),
                            'best_acc1': best_acc1,
                            'optimizer' : optimizer.state_dict(),
                        }, is_best, os.path.join(args.save_path, 'adv_trained_model'))
        

    #initial_acc, _ = validate(test_loader, model, criterion, 1, args)
    #import pdb
    #pdb.set_trace()

    #convert dataloader into an eagerPy tensor for FoolBox attack generation 
    adv_dict = gen_attacks(test_loader,
                           classifier, attack_list, epsilons, args.gpu_ids)

    #append cw attack if evaluated
    if 'carliniLinf' in args.attacks:
        adv_dict.update(cw_dict)

    # loop through all generated dataloaders with adversarial images
    results_dict = {}
    for attack_name in adv_dict:

        # measure attack success
        print("Testing performance of attack {}: ".format(attack_name))
        for epsilon_attack, epsilon in zip(adv_dict[attack_name], epsilons):
            attacked_acc, _ = validate(
                epsilon_attack, model, criterion, 1, args)

            # save adv images for visualization purposes
            dataiter = iter(epsilon_attack)
            images, _ = dataiter.next()
            img_grid = utils.make_grid(images)
            summary.add_image("Training Images Adversarially Attacked Using {} with eps {}".format(
                    attack_name, epsilon), img_grid)

            print("Generating defences for attack {} with eps {}: ".format(attack_name, epsilon))
            
            def_adv_dict = gen_defences(epsilon_attack, attack_name, defence_list)
            accuracies = {'initial': initial_acc.item(
            ), 'attacked': attacked_acc.item()}

            if 'adv_retraining' in args.defences:
                # evaluate retrained model 
                acc1, val_loss = validate(
                    epsilon_attack, robust_model, criterion, epoch, args)
                accuracies['adv_training'] = acc1.item()

            for def_name in def_adv_dict:
                print("Testing performance of defence {}: ".format(def_name))
                top1, _ = validate(def_adv_dict[def_name], model, criterion, 1, args)
                def_images , _ = zip(*[batch for batch in def_adv_dict[def_name]])
                def_images = torch.cat(def_images).numpy()

                accuracies[def_name] = top1.item()

                # save def images for visualization purposes
                dataiter = iter(def_adv_dict[def_name])
                images, _ = dataiter.next()
                img_grid = utils.make_grid(images)
                summary.add_image("Defense {} against Attack {} with eps {}".format(def_name, attack_name, epsilon), img_grid)

            results_dict[attack_name + ' eps {}'.format(epsilon)] = accuracies
        print(results_dict)
    
    with open(os.path.join(args.save_path, 'results.json'), 'w') as save_file:
        json.dump(results_dict, save_file)
