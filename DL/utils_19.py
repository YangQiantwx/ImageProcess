import torch, math
import torch.nn as nn
import numpy as np
from loss import *
from torchvision.utils import make_grid
# import argparse


# Define Loss function
    
criterionL1 = torch.nn.L1Loss()
criterionMSE = torch.nn.MSELoss()
criterionSSIM = SSIMLoss(device = "cuda")

def get_psnr(img, target):
    with torch.no_grad():
        mse = criterionMSE(img, target).cpu().item()
        psnr = 10 * math.log10(1 / mse)
    return psnr

# class Trainer():
#     def __init__(self, model, args, train_loader = None, valid_loader = None, lr_scheduler = None, max_steps = None):

def train(model, epoch, train_loader, optimizer, writer, log_step = 1):
    model.train()
    
    # global history_score
    avg_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterionL1(target, output) # + SSIMLoss()
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % log_step == 0 and batch_idx > 0:
            print('Train Epoch: {} [{}/{}]\tAveraged Loss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), avg_loss/batch_idx))
    writer.add_scalar('loss/epoch', avg_loss/batch_idx, epoch)
    return


def test(model, epoch, test_loader, writer, log_step = 5):
    model.eval()
    test_loss = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        # import pdb; pdb.set_trace()
        output = model(data)
        loss = criterionL1(output, target) 
        test_loss += loss.item()
        psnr = get_psnr(output, target)
    
    if epoch % log_step == 0: # only save the last image
        data = make_grid(data)
        writer.add_image('img_in', data, epoch)
        output = make_grid(output)
        writer.add_image('img_out', output, epoch)
        target = make_grid(target)
        writer.add_image('img_target', target, epoch)

    test_loss /= (batch_idx + 1)
    print("Average test loss: {:.4f}".format(test_loss))
    print("Average test psnr: {:.4f}".format(psnr))
    return test_loss, psnr