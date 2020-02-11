import os, shutil
import torch
import torchvision.utils as utils

# Below utilities adapted from: https://github.com/pytorch/examples/blob/master/imagenet/main.py

def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    filename=os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        model_pth = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, model_pth)

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

def save_images(dataset, save_dir, image_type, num_images=50):
    ''' Saves torch tensors from a torch dataloader as images in a 
    specified directory for visualization and sanity-checking
    
    inputs: 
    
    dataset (torch Dataloader)
    save_dir (directory path to save images in)
    image_type (string to specify what type of images (eg. clean, attacked, etc.))
    num_images (number of images from the Dataloader to save into the directory)

    output: none (simply saves images into save_dir)
    '''
    if not os.path.isdir(os.path.join(save_dir, 'images', image_type)):
        os.makedirs(os.path.join(save_dir, 'images', image_type))
    for i, image in enumerate(dataset[:num_images]):
        utils.save_image(torch.Tensor(image), os.path.join(save_dir, 'images', image_type, 'image_{}.png'.format(i)))
