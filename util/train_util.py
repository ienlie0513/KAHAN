import os
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from util.util import Progressor

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, warmup_epochs=10):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.warmup_epochs = warmup_epochs
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss, epoch):
        if epoch < self.warmup_epochs:
            return False
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def trainIters(model, trainset, validset, train, evaluate, epochs=100, learning_rate=0.01, weight_decay=1e-3, batch_size=32, save_info=None, print_every=1000, device='cuda', log=None):
    # plot every epoch
    plot_train_losses = []
    plot_test_losses = []
    plot_train_accs = []
    plot_test_accs = []

    train_acc = 0
    loss_total = 0
    
    max_acc = 0

    # Use DataParallel for multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    trainloader = data.DataLoader(trainset, batch_size, shuffle = True, pin_memory=True, num_workers=0)
    early_stopper = EarlyStopper(patience=6, min_delta=0.001)

    progress = Progressor('py', total=print_every, log=log)

    for i in range(epochs):
        train_acc = 0
        loss_total = 0
        
        for input_tensor, target_tensor in trainloader:
            loss, correct = train(input_tensor, target_tensor, model, optimizer, criterion, device)
            train_acc += correct
            loss_total += loss*len(input_tensor)
     
        # evaluate
        test_loss, test_acc, _, _ = evaluate(model, validset, device)

        # step the learning rate scheduler
        scheduler.step(test_loss)

        # save model
        if test_acc > max_acc:
            max_acc = test_acc
            model_name = save_model(model, save_info, test_acc, log)

        # plot every epoch
        train_acc /= len(trainset)
        plot_train_accs.append(train_acc)
        plot_test_accs.append(test_acc)
        plot_train_losses.append(loss_total/len(trainset))
        plot_test_losses.append(test_loss)

        progress.update(loss_total/len(trainset), test_loss, train_acc, test_acc, i)
        
        # reset tqdm bar
        if progress.count == print_every and i < (epochs-1):
            progress.reset(i)

        # # early stop
        # if early_stopper.early_stop(test_loss, i):
        #     tqdm.write("Early stop at epoch %s"%i)
        #     log.write("Early stop at epoch %s \n"%i)
        #     break
        
    tqdm.write("The highest accuracy is %s"%max_acc)
    log.write("The highest accuracy is %s\n"%max_acc)
            
    return plot_train_accs, plot_test_accs, plot_train_losses, plot_test_losses, model_name

def save_model(model, save_info, acc, log):
    fold, ckpt_dir = save_info
    now = datetime.now().strftime("%Y_%m_%d %H:%M:%S")
    model_name = "{}/model_{}.ckpt".format(ckpt_dir, fold)
    tqdm.write("Model {} saved at {} with acc: {:.4f}".format(model_name, now, acc))
    log.write("Model {} saved at {} with acc: {:.4f}\n".format(model_name, now, acc))
    torch.save(model.state_dict(), model_name)
    
    return model_name