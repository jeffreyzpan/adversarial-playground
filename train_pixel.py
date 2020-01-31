# Training code for PixelCNN++ adapted from https://github.com/pclucas14/pixel-cnn-pp

import time
import os
import shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
from tensorboardX import SummaryWriter
from models.pixel_utils import * 
from models.pixel_cnn import * 
#from PIL import Image

parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('--data_path', type=str,
                    default='datasets/', help='path to dataset')
parser.add_argument('--dataset', type=str, choices=['xrays', 'gtsrb'], help='choose dataset to train pixelcnn++ on')
parser.add_argument('--workers', type=int, default=16, help='number of data loading workers to use')
parser.add_argument('--pretrained', type=str, default='', help='path to pretrained model')
parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='comma-seperated string of gpu ids to use for acceleration (-1 for cpu only)')
# model checkpoint flags
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='checkpoints', help='Folder to save checkpoints and log.')
parser.add_argument('--train', action='store_true', help='train the model')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
# hyperparameters
parser.add_argument('--epochs', type=int, default=5000, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed to use')
args = parser.parse_args()

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if torch.cuda.is_available():
    gpu_id_list = [int(i.strip()) for i in args.gpu_ids.split(',')]
else:
    gpu_id_list = [-1]

if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)
writer = SummaryWriter(args.save_path)
print('==> Output path: {}...'.format(args.save_path))

rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5

if args.dataset == 'xrays':
    num_classes = 2
    input_shape = (1, 224, 224)
if args.dataset == 'gtsrb':
    num_classes = 43
    input_shape = (3, 28, 28)
    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)

train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.Resize((28, 28)), transforms.ToTensor()])
test_transform = transforms.Compose(
        [transforms.Resize((28, 28)), transforms.ToTensor()])
trainset = datasets.ImageFolder(os.path.join(args.data_path, args.dataset, 'train'), transform=train_transform)
testset = datasets.ImageFolder(os.path.join(args.data_path, args.dataset, 'val'), transform=test_transform) 

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_shape[0], nr_logistic_mix=args.nr_logistic_mix)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume) 
        checkpoint['state_dict'] = {n.replace('module.', '') : v for n, v in checkpoint['state_dict'].items()}
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'" .format(args.resume))
    else:
        print("=> No checkpoint found at '{}'".format(args.resume))
else:
    print("=> Not using any checkpoint for pixelCNN++ model")

if -1 not in gpu_id_list:
    model = torch.nn.DataParallel(model, device_ids = gpu_id_list)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

if -1 not in gpu_id_list and torch.cuda.is_available():
    model.cuda()

def sample(model):
    model.train(False)
    data = torch.zeros(args.batch_size, input_shape[0], input_shape[1], input_shape[2])
    data = data.cuda()
    for i in range(input_shape[1]):
        for j in range(input_shape[2]):
            with torch.no_grad():
                out   = model(data, sample=True)
            out_sample = sample_op(out)
            data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data

print('starting training')
writes = 0
min_loss = 2147483647
for epoch in range(args.epochs):
    model.train(True)
    torch.cuda.synchronize()
    train_loss = 0.
    time_ = time.time()
    model.train()
    for batch_idx, (input,_) in enumerate(train_loader):
        input = input.cuda()
        output = model(input)
        loss = loss_op(input, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data.item()
        if (batch_idx +1) % args.print_freq == 0 : 
            deno = args.print_freq * args.batch_size * np.prod(input_shape) * np.log(2.)
            writer.add_scalar('train/bpd', (train_loss / deno), writes)
            print('loss : {:.4f}, time : {:.4f}'.format(
                (train_loss / deno), 
                (time.time() - time_)))
            train_loss = 0.
            writes += 1
            time_ = time.time()
            

    # decrease learning rate
    scheduler.step()
    
    torch.cuda.synchronize()
    model.eval()
    test_loss = 0.
    with torch.no_grad():
        for batch_idx, (input,_) in enumerate(test_loader):
            input = input.cuda()
            output = model(input)
            loss = loss_op(input, output)
            test_loss += loss.data.item()
            del loss, output

        deno = batch_idx * args.batch_size * np.prod(input_shape) * np.log(2.)
        writer.add_scalar('test/bpd', (test_loss / deno), writes)
        print('test loss : %s' % (test_loss / deno))
    
    if test_loss < min_loss: 
        min_loss = test_loss
        filename =  os.path.join(args.save_path, 'checkpoint.pth.tar')
        torch.save(model.state_dict(), filename)
        model_pth = os.path.join(args.save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, model_pth)
        print('sampling...')
        sample_t = sample(model)
        sample_t = rescaling_inv(sample_t)
        utils.save_image(sample_t,'images/pixel_cnn_{}.png'.format(epoch), 
                nrow=5, padding=0)
