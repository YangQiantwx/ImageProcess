import os, sys, glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter   


from models.unet import UNet
from data.datasets import Dataset
from utils import train, test


# Training settings
parser = argparse.ArgumentParser(description='Encoder Decoder Pipeline')
parser.add_argument('--data_path_train', type=str, default='data/train/')
parser.add_argument('--data_path_test', type=str, default='data/test/')
parser.add_argument('--log', type=str, default='./log/')
parser.add_argument('--batch-size', type=int, default = 32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default = 50, metavar='N',
                    help='number of epochs to train the model')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
parser.add_argument('--save', default='checkpoint/', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--pretrain', default=True, type=bool, 
                    help='Load pretrain model')
parser.add_argument('--arch', default='Unet', type=str,
                    help='architecture to use')
parser.add_argument('--gpu', type=str, default='0,1', help='Select gpu to use')
parser.add_argument('--log_step', type=int, default='10', help='How many steps to print log')




if __name__=="__main__":
    
    args = parser.parse_args()
    print(args)

    # setup CUDA enviroment
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device_ids = []
    gpu_input = args.gpu.split(',')
    for i, id in enumerate(gpu_input):
        device_ids.append(int(i))
        print('Using GPU:', device_ids)
    torch.cuda.empty_cache()

    # Add ramdom to the training process
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    # Load data set
    train_dataset = Dataset(args.data_path_train) # tensor (batch_size = 16, channel = RGB 3, h = 224, w = 224)   
    test_dataset = Dataset(args.data_path_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False)

    # setup log and checkpoints
    os.makedirs(args.log, exist_ok=True)
    os.makedirs(args.save, exist_ok=True)
    writer = SummaryWriter(args.log)
    
    # Load Model
    if args.arch == "Unet":
        model = UNet(3, 3).cuda()
    else:
        print("The network is not supported!")
        sys.exit()