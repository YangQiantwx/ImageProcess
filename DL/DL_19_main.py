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
parser = argparse.ArgumentParser(description='Encoder_Decoder Training Pipeline')
parser.add_argument('--data_path_train', type=str, default='data/haze/train/')
parser.add_argument('--data_path_test', type=str, default='data/haze/test/')
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
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
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
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

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
        model = UNet(3, 3).cuda() # x = torch.zeros(16,3,224,224).cuda()
    else:
        print("The network is not supported!")
        sys.exit()

    if args.pretrain: # load saved model
        checkpoints = glob.glob(args.save + "*.pth")
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        ckpt = torch.load(latest_checkpoint)
        model.load_state_dict(ckpt)
        print("Load the pretrain model!")

    model = nn.DataParallel(model, device_ids = device_ids)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Train the model
    min_loss = 100.

    for epoch in range(args.epochs):
        train(model, epoch, train_loader, optimizer, writer, log_step = args.log_step)
        loss1, psnr1 = test(model, epoch, test_loader, writer)
        
        if isinstance(model, nn.DataParallel):
            model = model.module
        else:
            model = model
        
        print ('Best test accuracy/loss before the current epoch: {}'.format(min_loss))
        if loss1 < min_loss:
            min_loss = loss1
            torch.save(model.state_dict(), os.path.join(args.save, 'best.pth'))
        model = nn.DataParallel(model, device_ids = device_ids)
        print('test pnsr: ', psnr1)
        print ('Best test loss: {}'.format(min_loss))
        print ('End of {}th epoch\n'.format(epoch))
    print('End!')