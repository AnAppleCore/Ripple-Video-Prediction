import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from model import RippleHeight
from dataset import get_dataloader
from utils import train, validate, predict, plot_curves

def get_args():
    parser = argparse.ArgumentParser('Hyperparameters setting')
    parser.add_argument('-e', '--epochs', type = int, default = 200)
    parser.add_argument('-b', '--batch_size', type = int, default = 4)
    parser.add_argument('-m', '--momentum', type = float, default = 0.9)
    parser.add_argument('-w', '--weight_decay', type = float, default = 1e-4)
    parser.add_argument('-r', '--learning_rate', type = float, default = 1e-2)
    parser.add_argument('-p', '--predict', action='store_true')
    parser.add_argument('--weights', type = str, default = None)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--frame_cnt', type = int, default = 100)
    parser.add_argument('--pin_memory', type = bool, default = True)
    parser.add_argument('--img_name', type = str, default = '0.png')
    parser.add_argument('--save_path', type = str, default = './output')
    parser.add_argument('--img_shape', nargs='+', type=int, default=[256, 256])
    args = parser.parse_args()
    print('Arguments:', args)
    # Create results directory
    if not os.path.isdir(args.save_path+'/videos/'+args.img_name.split('.')[0]):
        os.makedirs(args.save_path+'/videos/'+args.img_name.split('.')[0])
    if not os.path.isdir(args.save_path+'/images/'+args.img_name.split('.')[0]):
        os.makedirs(args.save_path+'/images/'+args.img_name.split('.')[0])
    return args

def main():
    args = get_args()

    # data loading
    train_loader = get_dataloader(args)

    # model initialization
    model = RippleHeight(in_channels=3, out_channels=1)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # patameters and metrics initialization
    best_acc = 0.0
    start_epoch = 0
    metrics = {
        'train_loss': [],
        'train_acc': []
    }
    best_path = args.save_path + '/best_model.pth.tar'

    # load checkpoint
    if args.weights is not None:
        print('loade checkpoint from {}'.format(args.weights))
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
        best_acc = checkpoint['best_acc']
        metrics = checkpoint['metrics']

    # push model to GPU
    if torch.cuda.is_available():
        model = model.cuda()
        print('push model to #{} GPU: {}'.format(torch.cuda.current_device, torch.cuda.get_device_name))
    else:
        print('No GPU available.')
        return

    # generating log file
    with open(args.save_path + '/model_log.csv', 'a') as log:
        log.write('epoch, train_loss, train_acc\n')

    start = time.time()
    for epoch in range(start_epoch, args.epoch):

        # trainning
        print('------Training------')
        train_loss, train_acc = train(train_loader, model, loss_func, optimizer, scheduler, epoch, args)

        # update metrics
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        print('Train epoch {} complete! train loss: {:.4f}, acc {:.4f}'.format(epoch, train_loss, train_acc))

        # write log
        with open(args.save_path + '/model_log.csv', 'a') as log:
            log.write('{}, {:.5f}, {:.5f}\n'.format(epoch, train_loss, train_acc))

        # save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'metrics': metrics,
            'epoch': epoch,
        }, args.save_path + '/checkpoint.pth.tar')

        # save best model
        if train_acc > best_acc:
            print('train acc improved from {:4f} to {:4f}.'.format(best_acc, train_acc))
            best_acc = train_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict()
            }, best_path)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # plot_curves(metrics, args)

if __name__ == '__main__':
    main()