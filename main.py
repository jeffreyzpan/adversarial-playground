import os, sys, shutil, time, random
import argparse

import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import models

import art.attacks.evasion as evasion
import art.defences as defences

model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Adversarial Training Benchmarking', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='datasets/', help='path to dataset')
parser.add_argument('--dataset', type=str, choices=['sms-spam', 'xrays', 'gtsrb', 'toxic-comments'], help='choose dataset to benchmark adversarial training techniques on.')
parser.add_argument('--arch', metavar='ARCH', default='resnet50', help='model architecture: to evaluate robustness on (default: resnet50)')
parser.add_argument('--workers', type=int, default=16, help='number of data loading workers to use')
parser.add_argument('--pretrained', type=str, default='', help='path to pretrained model')
parser.add_argument('--gpu', action='store_true', help='use gpu')
# Hyperparameters
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')

# Model checkpoint flags
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='checkpoints', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

# Experiments
parser.add_argument('--attacks', type=str, default='all', help='comma seperated string of attacks to evaluate')
parser.add_argument('--defences', type=str, default='all', help='comma seperated string of defences to evaluate')


def main():
    args = parser.parse_args()

    assert args.arch in model_names, 'Error: model {} not supported'.format(args.arch)

    if args.dataset == 'xrays' or args.dataset == 'sms-spam':
        num_classes = 2
    if args.dataset == 'gtsrb':
        num_classes = 43
    if args.dataset == 'toxic-comments':
        num_classes = 6

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.Resize((224, 224)), transforms.ToTensor()])
    test_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()])

    trainset = dset.ImageFolder(os.path.join(args.data_path, args.dataset, 'train'), transform=train_transform)
    testset = dset.ImageFolder(os.path.join(args.data_path, args.dataset, 'val'), transform=test_transform) 

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = models.__dict__[args.arch]()
    print(model)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                    weight_decay=args.weight_decay, nesterov=True)

    if torch.cuda.is_available():
        net.cuda()
        criterion.cuda()

    attack_name_list = args.attacks.split(',')
    attack_name_list = [i.strip().lower() for i in attack_name_list] #sanitize inputs

    defence_name_list = args.defences.split(',')
    defence_name_list = [i.strip().lower() for i in defence_name_list] #sanitize inputs

    attack_list = {}
    defence_list = {}

    #initialize attacks and append to dict

    if 'fgsm' in attack_name_list:
        attack_list['fgsm'] = evasion.FastGradientMethod(classifier)
    if 'pgd' in attack_name_list:
        attack_list['pgd'] = evasion.ProjectedGradientDescent(classifier)
    if 'hopskipjump' in attack_name_list:
        attack_list['hsj'] = evasion.HopSkipJump(classifier)
    if 'query-efficient' in attack_name_list:
        raise NotImplementedError
    if 'deepfool' in attack_name_list:
        attack_list['deepfool'] = evasion.DeepFool(classifier)

    #initialize defenses and append to dict

    if 'thermometer' in defence_name_list:
        defence_list['thermometer'] = defences.ThermometerEncoding(clip_values) #TODO figure out what actually should go here 
    if 'pixeldefend' in defence_name_list:
        defence_list['pixeldefend'] = defences.PixelDefend(clip_values) #TODO figure out what goes here
    if 'tvm' in defence_name_list:
        defence_list['tvm'] = defences.TotalVarMin()
    if 'saddlepoint' in defence_name_list:
        defence_list['saddlepoint'] = defences.AdversarialTrainer(classifier, attacks=attack_list['pgd'])

    # Below training loop, train function, val function, and utilties for those functions are adapted from: https://github.com/pytorch/examples/blob/master/imagenet/main.py

    for epoch in range(args.epochs):

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        
        save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best)

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
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
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

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

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    if args.dataset != 'xrays':
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, losses, top1, top5],
            prefix="Test: ")
    else:
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, losses, top1],
            prefix="Test: ")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

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

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()