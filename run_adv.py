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
from lib.datasets.data_utils import generate_dataset
from lib.adversarial.adversarial import *

from art.classifiers import PyTorchClassifier
import art.attacks.evasion as evasion
import art.defences as defences
import numpy as np

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
parser.add_argument('--fold', type=int,
                    help='if evaluating urbansound dataset, fold number to use for validation')
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
                    help='number of epochs to train for')
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
                    'fgsm', 'carliniL2', 'pgd', 'deepfool', 'hopskipjump', 'pixelattack'], help='list of attacks to evaluate')
parser.add_argument('--defences', type=str, nargs='+', default=[], help='list of defences to evaluate')

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
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # check that CUDA is actually available and pass in GPU ids, use CPU if not
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

    assert args.arch in model_names, 'Error: model {} not supported'.format(
        args.arch)

    # set variables based on dataset to evaluate on
    if args.dataset == 'imagenet':
        input_size = 224
    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
        input_size = 32
    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        input_size = 28

    criterion = torch.nn.CrossEntropyLoss()
    train_loader, test_loader, num_classes = generate_dataset(
        args.dataset, args.data_path, input_size, args.batch_size, args.workers)

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
        model.cuda()
        criterion.cuda()

    cudnn.benchmark = True

    # perform attacks and defences on dataset

    attack_list = {}
    defence_list = {}

    # initialize attacks and append to dict

    classifier = classifier = PyTorchClassifier(model=copy.deepcopy(model), clip_values=(
        0, 1), loss=criterion, optimizer=optimizer, input_shape=(3, input_size, input_size), nb_classes=num_classes)

    with open('parameters/{}_parameters.json'.format(args.dataset)) as f:
        parameter_list = json.load(f)

    if 'fgsm' in args.attacks:
        fgsm_params = parameter_list['fgsm']
        attack_list['fgsm'] = evasion.FastGradientMethod(classifier, targeted=fgsm_params['targeted'],
                eps=fgsm_params['eps'], minimal=fgsm_params['minimal'], batch_size=fgsm_params['batch_size'])
    if 'carliniL2' in args.attacks:
        carlini_params = parameter_list['carliniL2']
        attack_list['carliniL2'] = evasion.CarliniL2Method(classifier, confidence=carlini_params['confidence'], targeted=carlini_params['targeted'],
                learning_rate=carlini_params['learning_rate'], binary_search_steps=carlini_params['binary_search_steps'],
                max_iter=carlini_params['max_iter'], initial_const=carlini_params[
                    'initial_const'], max_halving=carlini_params['max_halving'],
                max_doubling=carlini_params['max_doubling'], batch_size=carlini_params['batch_size'])
    if 'pgd' in args.attacks:
        pgd_params = parameter_list['pgd']
        attack_list['pgd'] = evasion.ProjectedGradientDescent(classifier, targeted=pgd_params['targeted'],
                max_iter=pgd_params['max_iter'], eps_step=pgd_params['eps_step'], eps=pgd_params['eps'], batch_size=pgd_params['batch_size'])

    if 'deepfool' in args.attacks:
        fool_params = parameter_list['deepfool']
        attack_list['deepfool'] = evasion.DeepFool(classifier, epsilon=fool_params['epsilon'],
                max_iter=fool_params['max_iter'], batch_size=fool_params['batch_size'], nb_grads=fool_params['nb_grads'])
    if 'hopskipjump' in args.attacks:
        hsj_params = parameter_list['hopskipjump']
        attack_list['hopskipjump'] = evasion.HopSkipJump(classifier, max_iter=hsj_params['max_iter'], max_eval=hsj_params['max_eval'],
                init_eval=hsj_params['init_eval'], init_size=hsj_params['init_size'], targeted=hsj_params['targeted'])
    if 'pixelattack' in args.attacks:
        pixelattack_params = parameter_list['pixelattack']
        attack_list['pixelattack'] = evasion.PixelAttack(classifier, th=pixelattack_params['th'],
                es=pixelattack_params['es'], targeted=pixelattack_params['targeted'], verbose=False)

     # initialize defenses and append to dict

    if 'pixeldefend' in args.defences:
        pixel_params = parameter_list['pixeldefend']
        defence_list['pixeldefend'] = defences.PixelDefend(clip_values=(
            pixel_params['clip_min'], pixel_params['clip_min']), eps=pixel_params['eps'])
    if 'tvm' in args.defences:
        tvm_params = parameter_list['tvm']
        defence_list['tvm'] = defences.TotalVarMin(clip_values=(
            tvm_params['clip_min'], tvm_params['clip_max']), prob=tvm_params['prob'], lamb=tvm_params['lamb'], max_iter=tvm_params['max_iter'])
    if 'jpeg' in args.defences:
        jpeg_params = parameter_list['jpeg']
        defence_list['jpeg'] = defences.JpegCompression(clip_values=(
            jpeg_params['clip_min'], jpeg_params['clip_max']), channel_index=jpeg_params['channel_index'], quality=jpeg_params['quality'])
    if 'i_defender' in args.defences:
        model.module.insert_forward_hooks(input_shape, cuda=True)
        i_params = parameter_list['i_defender']
        defense_model = models.__dict__['i_defender'](model, train_loader, num_classes, i_params['p_value'],
                                                      n_components=i_params['n_components'], max_iter=i_params['max_iter'], n_init=i_params['n_init'])
        model.module.remove_hooks()
    if 'thermometer' in args.defences:
        thermometer_params = parameter_list['thermometer']
        defence_list['thermometer'] = defences.ThermometerEncoding(clip_values=(
            thermometer_params['clip_min'], thermometer_params['clip_max']), num_space=thermometer_params['num_space'], channel_index=thermometer_params['channel_index'])

    if 'distillation' in args.defences:
        distillation_params = parameter_list['distillation']
        defence_list['distillation'] = defences.transformer.DefensiveDistillation(
            classifier, batch_size=distillation_params['batch_size'], nb_epochs=distillation_params['epochs'])

    # get initial validation set accuracy

    initial_acc, _ = validate(test_loader, model, criterion, 1, args)

    # ART appears to only support numpy arrays, so convert dataloader into a numpy array of images
    image_batches, label_batches = zip(*[batch for batch in test_loader])
    test_images = torch.cat(image_batches).numpy()
    test_labels = torch.cat(label_batches).numpy()
    adv_dict = gen_attacks(test_images, test_labels,
                           classifier, criterion, attack_list)

    # loop through all generated dataloaders with adversarial images
    results_dict = {}
    for attack_name in adv_dict:

        # measure attack success
        print("Testing performance of attack {}: ".format(attack_name))
        attacked_acc, _ = validate(
            adv_dict[attack_name], model, criterion, 1, args)

        adv_images, adv_labels = zip(
            *[batch for batch in adv_dict[attack_name]])
        adv_images = torch.cat(adv_images).numpy()
        adv_labels = torch.cat(adv_labels).numpy()

        # save adv images for visualization purposes
        dataiter = iter(adv_dict[attack_name])
        images, _ = dataiter.next()
        img_grid = utils.make_grid(images)
        summary.add_image("Training Images Adversarially Attacked Using {}".format(
                attack_name), img_grid)

        print("Generating defences for attack {}: ".format(attack_name))

        def_adv_dict = gen_defences(
            test_images, adv_images, attack_name, test_labels, classifier, criterion, defence_list)
        accuracies = {'initial': initial_acc.item(
        ), 'attacked': attacked_acc.item()}

        if 'i_defender' in args.defences:
            model.module.update_defender(defense_model)
            validate(adv_dict[attack_name], model, criterion, 1, args)
            raw_attack_log = model.module.fetch_attack_log()
            attack_log_probs = model.module.fetch_raw_probs()
            # flatten attack log into 1D list
            attack_log = np.array(
                [j for sub_list in raw_attack_log for j in sub_list])
            attack_log_probs = np.array(
                [j for sub_list in attack_log_probs for j in sub_list])
            validate(test_loader, model, criterion, 1, args)
            raw_clean_log = model.module.fetch_attack_log()
            clean_log_probs = model.module.fetch_raw_probs()
            # flatten attack log into 1D list
            clean_log = np.array(
                [j for sub_list in raw_clean_log for j in sub_list])
            clean_log_probs = np.array(
                [j for sub_list in clean_log_probs for j in sub_list])
            attack_num = sum(attack_log)
            clean_num = sum(clean_log)
            accuracies['i_defender_attacked'] = float(
                attack_num/len(adv_dict[attack_name].dataset))
            accuracies['i_defender_clean'] = float(
                1-clean_num/len(test_loader.dataset))

        if 'adv_training' in args.defences or 'thermometer' in args.defences:
            # attack training set images with attack and save it to a dataloader
            if 'adv_training' in args.defences:
                print('Generating adversarial examples on training data')
                adv_loader = adversarial_retraining(
                    train_loader, attack_list[attack_name])
                robust_model = copy.deepcopy(model)
                epochs = 20
                top1 = 0
                for epoch in range(epochs):
                    train(adv_loader, robust_model,
                          criterion, optimizer, epoch, args)
                    acc1, val_loss = validate(
                        adv_dict[attack_name], robust_model, criterion, epoch, args)
                    top1 = max(top1, acc1)
                accuracies['adv_training'] = acc1.item()

            if 'thermometer' in args.defences:

                clean_encoded_loader, adv_encoded_loader = thermometer_encoding(
                    train_loader, adv_dict[attack_name], parameter_list['thermometer'])

                enc_model = copy.deepcopy(model)
                epochs = 20
                top1 = 0

                for epoch in range(epochs):
                    train(clean_encoded_loader, enc_model,
                          criterion, optimizer, epoch, args)
                    acc1, val_loss = validate(
                        adv_encoded_loader, enc_model, criterion, epoch, args)
                    top1 = max(top1, acc1)
                accuracies['thermometer'] = acc1.item()

        for def_name in def_adv_dict:
            print("Testing performance of defence {}: ".format(def_name))
            top1, _ = validate(def_adv_dict[def_name], model, criterion, 1, args)
            def_images , _ = zip(*[batch for batch in def_adv_dict[def_name]])
            def_images = torch.cat(def_images).numpy()

            accuracies[def_name] = top1.item()

            # save def images for visualization purposes
            if args.dataset == 'xrays' or args.dataset == 'gtsrb':
                dataiter = iter(def_adv_dict[def_name])
                images, _ = dataiter.next()
                img_grid = utils.make_grid(images)
                summary.add_image("Defense {} against Attack {}".format(def_name, attack_name), img_grid)

        results_dict[attack_name] = accuracies
    print(results_dict)
    
    with open(os.path.join(args.save_path, 'results.json'), 'w') as save_file:
        json.dump(results_dict, save_file)
