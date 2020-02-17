import time
import datetime
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import network
import dataset
import utils

# This code is not run
def Trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # cudnn benchmark accelerates the network
    if opt.cudnn_benchmark == True:
        cudnn.benchmark = True
    else:
        cudnn.benchmark = False

    # VGG16 network
    net = utils.create_network(opt)

    # To device
    if opt.multi_gpu == True:
        net = nn.DataParallel(net)
        net = net.cuda()
    else:
        net = net.cuda()

    # Loss functions
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # Optimizers
    optimizer = torch.optim.SGD(net.parameters(), lr = opt.lr, momentum = opt.momentum, weight_decay = opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(optimizer, epoch, opt):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = opt.lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), '%s_%s_epoch%d_bs%d.pth' % (opt.sub_type, opt.task, epoch, opt.batch_size))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), '%s_%s_epoch%d_bs%d.pth' % (opt.sub_type, opt.task, epoch, opt.batch_size))
        print('The trained model is successfully saved at epoch %d' % epoch)
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.ImageNetTrainSet(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()

    # Training loop
    for epoch in range(opt.epochs):
        for batch_idx, (data, target) in enumerate(dataloader):

            # Load data and put it to cuda
            data = data.cuda()
            target = target.cuda()

            # Train one iteration
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Cross-Entropy Loss: %.5f] time_left: %s" %
                ((epoch + 1), opt.epochs, batch_idx, len(dataloader), loss.item(), time_left))

        # Learning rate decrease
        adjust_learning_rate(optimizer, (epoch + 1), opt)

        # Save the model
        save_model(net, (epoch + 1), opt)
