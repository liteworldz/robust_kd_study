#from torchvision import transforms
import torch

from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
import warnings
import torchattacks
#from torchvision import models
from resnet import *
from wideresnet import WideResNet
#from preact_resnet import *
#from models import *
from utils import progress_bar
import argparse
import torch.backends.cudnn as cudnn

import dataset_npz
import dataset_npz_100
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from loss import LossCalulcator

import time
import csv


from pytorchtools import EarlyStopping

import os
import sys

DEVICES_IDS = [0]


'''
DEFAULT CONSTANTS
'''
EVAL_INDEX = 0  # after this epoch index will start evaluating model during training 
EPS = 8/255
ALPHA = 2/255
STEPS = 10
STANDARD = 'nt'
ADVERSARIAL = 'at'
KDISTILLATION = 'kd'
ALPISTILLATION = 'alp'
TEACHER = 'NT'
STUDENT = 'KD'
ADV = 'AT'
ALP = 'ALP'
MOMENTUM = 0.9

def get_model_by_name(name, num_classes):
    if name == "wideresnet":
        DECAY = 5e-4
        model = WideResNet(28, num_classes=num_classes, widen_factor=10, dropRate=0.0)
    elif name == "resnet18":
        DECAY = 2e-4
        model = ResNet18(num_classes=num_classes) #models.resnet18() # 
    else:
        raise Exception('Unknown network name: {0}'.format(name))
    return model, DECAY

             
def Train(logname, net, DECAY, network, classes, batch_size, train_size, train_dataset, val_dataset, beta, cutmix_prob,
             nb_epochs=10, learning_rate=0.1, patience=200, VERSION='_v1'):
    net.train()
    start = time.time()

    optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                          momentum=MOMENTUM, weight_decay=DECAY) 
    loss_func = nn.CrossEntropyLoss()
    train_loss = []
    total = 0
    correct = 0
    step = 0
    acc = 0
    best_acc = 0
    
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # Define the indices for the entire dataset
    indices = list(range(train_size))    
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience, verbose=True) 
    # breakstep = 0
    print("Standard Training (Benign) Started..")
    if beta > 0 :
        print('\n=> CutMix')
    for _epoch in range(nb_epochs):
        optimizer, lr = adjust_learning_rate(learning_rate, optimizer, _epoch)
        print(f'Epoch: {_epoch + 1} ' + 
              f'acc: {acc:.3f} ' +
              f'Elapsed Time (Min): {int(np.floor((time.time() - start)/60))} ' + 
              f'lr: { lr:.4f}')
        net.train()
        

        # Define the sampler for each epoch
        sampler = SubsetRandomSampler(indices[:60000]) 
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        for xs, ys in train_loader:
            xs, ys = Variable(xs), Variable(ys)
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()            
            
            '''
            CUT-MIX
            '''
            r = np.random.rand(1)
            if beta > 0 and r < cutmix_prob:
                #print('=> cutmix')
                # generate mixed sample
                lam = np.random.beta(beta, beta)
                rand_index = torch.randperm(xs.size()[0]).cuda()
                target_a = ys
                target_b = ys[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(xs.size(), lam)
                xs[:, :, bbx1:bbx2, bby1:bby2] = xs[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (xs.size()[-1] * xs.size()[-2]))
                # compute output
                preds =  net(xs) 
                loss = loss_func(preds, target_a) * lam + loss_func(preds, target_b) * (1. - lam)
                preds_np = preds.cpu().detach().numpy()
                correct += (np.argmax(preds_np, axis=1) ==
                            target_a.cpu().detach().numpy()).sum()
            else:
                preds = net(xs)
                loss = loss_func(preds, ys)
                _, predicted = torch.max(preds, 1)
                correct += (predicted == ys).sum().item()
            #correct += predicted.eq(ys).sum().item()
            train_losses.append(loss.data.item()) # record training loss
            total += xs.size(0)
            step += 1
            optimizer.zero_grad()
            loss.backward()  # calc gradients
            optimizer.step()  # update gradients
            #if total % 1000 == 0:
        acc = float(correct) / total
        print('[%s] Training accuracy: %.2f%%' %
                (step, acc * 100))
        total = 0
        correct = 0
        valid_loss = 0.0
        val_acc = 0.0
        if _epoch >=EVAL_INDEX:
            valid_losses, val_acc = evalClean(net, val_loader)
            valid_loss = np.average(valid_losses)
            avg_valid_losses.append(valid_loss)
        #scheduler.step()
        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)
        epoch_len = len(str(nb_epochs))
        
        print_msg = (f'[{_epoch + 1:>{epoch_len}}/{nb_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' +
                     f'Train Acc: {acc:.5f} ' +
                     f'Val Acc: {val_acc:.5f}')
        
        print(print_msg)
        
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'val_acc': val_acc,
            'network':network,
            'classes':classes,
            'epoch': _epoch + 1
                    }
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(_epoch+1, val_acc, net, TEACHER + '_' + VERSION, state)
        with open(logname, 'a', newline='') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([f'{_epoch + 1}', f'{train_loss:.6f}', f'{valid_loss:.6f}', f'{acc:.3f}', f'{val_acc:.3f}', f'{lr:.4f}', 'NT', int(early_stopping.counter), f'{int(np.floor((time.time() - start)/60))}'])
            
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        
def advTrain(logname, net, DECAY, network, classes, batch_size, train_size, train_dataset, val_dataset, beta, cutmix_prob,
             nb_epochs=10, learning_rate=0.1, patience=200, VERSION='_v1'):
    net.train()
    start = time.time()

    optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                          momentum=MOMENTUM, weight_decay=DECAY) 
    loss_func = nn.CrossEntropyLoss()
    train_loss = []
    total = 0
    correct = 0
    step = 0
    acc = 0
    best_acc = 0
    
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # Define the indices for the entire dataset
    indices = list(range(train_size))    
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience, verbose=True) 
    # breakstep = 0
    print("Adversarial Training (Robust) Started..")
    attack = torchattacks.PGD(net, eps=EPS, alpha=ALPHA, steps=STEPS)
    if beta > 0 :
        print('\n=> CutMix')
    for _epoch in range(nb_epochs):
        optimizer, lr = adjust_learning_rate(learning_rate, optimizer, _epoch)
        print(f'Epoch: {_epoch + 1} ' + 
              f'acc: {acc:.3f} ' +
              f'Elapsed Time (Min): {int(np.floor((time.time() - start)/60))} ' + 
              f'lr: { lr:.4f}')
        net.train()
        
        
        # Define the sampler for each epoch
        sampler = SubsetRandomSampler(indices[:60000]) 
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        for xs, ys in train_loader:
            xs, ys = Variable(xs), Variable(ys)
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()            
            
            '''
            CUT-MIX
            '''
            # Apply CutMix with beta=1.0 for first 100 epochs, then decrease beta linearly to 0 for remaining epochs
            #beta = 1.0 if _epoch <= 100 else 1.0 - (_epoch-100)/100
            r = np.random.rand(1)
            if beta > 0 and r < cutmix_prob:
                #print('=> cutmix')
                # generate mixed sample
                lam = np.random.beta(beta, beta)
                rand_index = torch.randperm(xs.size()[0]).cuda()
                target_a = ys
                target_b = ys[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(xs.size(), lam)
                xs[:, :, bbx1:bbx2, bby1:bby2] = xs[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (xs.size()[-1] * xs.size()[-2]))
                # compute output
                adv_a = attack(xs, target_a)
                adv_b = attack(xs, target_b)
                preds_a =  net(adv_a)
                preds_b =  net(adv_b) 
                loss = loss_func(preds_a, target_a) * lam + loss_func(preds_b, target_b) * (1. - lam)
                preds_np_a = preds_a.cpu().detach().numpy()
                preds_np_b = preds_b.cpu().detach().numpy()
                correct += (np.argmax(preds_np_a, axis=1) ==
                            target_a.cpu().detach().numpy()).sum()
                correct += (np.argmax(preds_np_b, axis=1) ==
                            target_b.cpu().detach().numpy()).sum()
            else:
                adv = attack(xs, ys)
                preds = net(adv)
                loss = loss_func(preds, ys)
                _, predicted = torch.max(preds, 1)
                correct += (predicted == ys).sum().item()
            #correct += predicted.eq(ys).sum().item()
            train_losses.append(loss.data.item()) # record training loss
            total += xs.size(0)
            step += 1
            optimizer.zero_grad()
            loss.backward()  # calc gradients
            optimizer.step()  # update gradients
            #if total % 1000 == 0:
        acc = float(correct) / total
        print('[%s] Adv Training accuracy: %.2f%%' %
                (step, acc * 100))
        total = 0
        correct = 0
        valid_loss = 0.0
        val_acc = 0.0
        if _epoch >=EVAL_INDEX:
            valid_losses, val_acc = evalAdvAttack(net, val_loader)
            valid_loss = np.average(valid_losses)
            avg_valid_losses.append(valid_loss)
        #scheduler.step()
        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)
        epoch_len = len(str(nb_epochs))
        
        print_msg = (f'[{_epoch + 1:>{epoch_len}}/{nb_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' +
                     f'Train Acc: {acc:.5f} ' +
                     f'Val Acc: {val_acc:.5f}')
        
        print(print_msg)
        
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'val_acc': val_acc,
            'network':network,
            'classes':classes,
            'epoch': _epoch + 1
                    }
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(_epoch+1, val_acc, net, ADV + '_' + VERSION, state)
        with open(logname, 'a', newline='') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([f'{_epoch + 1}', f'{train_loss:.6f}', f'{valid_loss:.6f}', f'{acc:.3f}', f'{val_acc:.3f}', f'{lr:.4f}', 'AT', int(early_stopping.counter), f'{int(np.floor((time.time() - start)/60))}'])
            
        if early_stopping.early_stop:
            print("Early stopping")
            break


def advALPTrain(logname, net, DECAY, network, classes, batch_size, train_size, device, train_dataset, val_dataset, beta, cutmix_prob,
               nb_epochs=10, distillation_weight=0.5, temperature=1, training_loss='alp', learning_rate=0.1, patience=200, VERSION='v1'):
    net.train()
    start = time.time()
    
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=MOMENTUM, weight_decay=DECAY)
    train_loss = []
    total = 0
    correct = 0
    step = 0
    acc = 0
    best_acc = 0
    log = []
    CELoss = nn.CrossEntropyLoss()
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # Define the indices for the entire dataset
    indices = list(range(train_size))    
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience, verbose=True)
    
    print('==> Training will run: ', nb_epochs, ' epochs')
    print("Adversarial ALP Training (Robust-ALP) Started..")
    attack = torchattacks.PGD(net, eps=EPS, alpha=ALPHA, steps=STEPS)
    if beta > 0 :
        print('\n=> CutMix')
    for _epoch in range(nb_epochs):
        optimizer, lr = adjust_learning_rate(learning_rate, optimizer, _epoch)
        lossCalculator = LossCalulcator(
            temperature, distillation_weight, _epoch, training_loss).to(device, non_blocking=True)
        print(f'Epoch: {_epoch + 1} ' + 
              f'acc: {acc:.3f} ' +
              f'Elapsed Time (Min): {int(np.floor((time.time() - start)/60))} ' + 
              f'lr: { lr:.4f}')
        net.train()
        
        
        # Define the sampler for each epoch
        sampler = SubsetRandomSampler(indices[:60000]) 
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        for xs, ys in train_loader:
            xs, ys = Variable(xs), Variable(ys)
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()            
            
            '''
            CUT-MIX
            '''
            r = np.random.rand(1)
            if beta > 0 and r < cutmix_prob:
                #print('=> cutmix')
                # generate mixed sample
                lam = np.random.beta(beta, beta)
                rand_index = torch.randperm(xs.size()[0]).cuda()
                target_a = ys
                target_b = ys[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(xs.size(), lam)
                xs[:, :, bbx1:bbx2, bby1:bby2] = xs[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (xs.size()[-1] * xs.size()[-2]))
                # compute output
                adv_a = attack(xs, target_a)
                adv_b = attack(xs, target_b)
                preds_t = net(xs)
                preds_a =  net(adv_a)
                preds_b =  net(adv_b) 
                #printLogits(outputs= preds_a, teacher=preds_t, targets= target_a, index=3, message= 'preds_a', escape=False)
                #printLogits(outputs= preds_b, teacher=preds_t, targets= target_b, index=3, message= 'preds_b', escape=True)
                loss = lossCalculator(benign=preds_t, targets=target_a, adversarial=preds_a, teacher=preds_t) * lam + lossCalculator(benign=preds_t, targets=target_b, adversarial=preds_b, teacher=preds_t) * (1. - lam)
                preds_np_a = preds_a.cpu().detach().numpy()
                preds_np_b = preds_b.cpu().detach().numpy()
                correct += (np.argmax(preds_np_a, axis=1) ==
                            target_a.cpu().detach().numpy()).sum()
                correct += (np.argmax(preds_np_b, axis=1) ==
                            target_b.cpu().detach().numpy()).sum()
            else:
                # compute output
                adv = attack(xs, ys)
                preds_t = net(xs)
                preds =  net(adv)
                loss = lossCalculator(benign=preds, targets=ys, adversarial=preds_t, teacher=preds_t)
                _, predicted = torch.max(preds, 1)
                correct += (predicted == ys).sum().item()
                #printLogits(outputs= preds, teacher=preds_t, targets= ys, index=0, message= 'preds', escape=False)
            train_losses.append(loss.data.item()) # record training loss
        
            total += xs.size(0)

            step += 1
            optimizer.zero_grad()
            loss.backward()  # calc gradients
            optimizer.step()  # update gradients
            #if total % 1000 == 0
        acc = float(correct) / total
        print('[%s] Adv ALP Training accuracy: %.2f%%' %
                (step, acc * 100))
        total = 0
        correct = 0
        log = lossCalculator.get_log()
        valid_loss = 0.0
        val_acc = 0.0
        if _epoch >=EVAL_INDEX:
            valid_losses, val_acc = evalAdvAttack(net, val_loader)
            valid_loss = np.average(valid_losses)
            avg_valid_losses.append(valid_loss)
        #scheduler.step()
        train_loss = np.average(train_losses)    
        avg_train_losses.append(train_loss)
        epoch_len = len(str(nb_epochs))
        
        print_msg = (f'[{_epoch + 1:>{epoch_len}}/{nb_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' +
                     f'Train Acc: {acc:.5f} ' +
                     f'Val Acc: {val_acc:.5f}')
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'val_acc': val_acc,
            'network':network,
            'classes':classes,
            'epoch': _epoch + 1
                    }
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(_epoch+1,val_acc, net, VERSION + '_' + training_loss + '_A_' + str(distillation_weight) + '_T_' + str(temperature) , state)
        with open(logname, 'a', newline='') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([f'{_epoch + 1}', f'{train_loss:.6f}', f'{valid_loss:.6f}', f'{acc:.3f}', f'{val_acc:.3f}', f'{lr:.4f}', training_loss, log, distillation_weight, temperature, int(early_stopping.counter), f'{int(np.floor((time.time() - start)/60))}' ])
            
        if early_stopping.early_stop:
            print("Early stopping")
            break
   

def advKDTrain(logname, net, DECAY, network, classes, batch_size, train_size, net_t, device, train_dataset, val_dataset, beta, cutmix_prob,
               nb_epochs=10, distillation_weight=0.5, temperature=1, training_loss='nt', learning_rate=0.1, patience=200, VERSION='v1'):
    net.train()
    net_t.eval()
    start = time.time()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=MOMENTUM, weight_decay=DECAY) 
    train_loss = []
    total = 0
    correct = 0
    step = 0
    acc = 0
    best_acc = 0
    log = []
    CELoss = nn.CrossEntropyLoss()

    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # Define the indices for the entire dataset
    indices = list(range(train_size))    
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience, verbose=True)
    
    print('==> Training will run: ', nb_epochs, ' epochs')
    print("Adversarial KD Training (Robust-KD) Started..")
    if beta > 0 :
        print('\n=> CutMix')
    attack = torchattacks.PGD(net, eps=EPS, alpha=ALPHA, steps=STEPS)
    #attack = torchattacks.PGD(net, eps=EPS, alpha=ALPHA, steps=STEPS)
    for _epoch in range(nb_epochs):
        lossCalculator = LossCalulcator(
            temperature, distillation_weight, _epoch, training_loss).to(device, non_blocking=True)
        optimizer, lr = adjust_learning_rate(learning_rate, optimizer, _epoch)
        print(f'Epoch: {_epoch + 1} ' + 
              f'acc: {acc:.3f} ' +
              f'Elapsed Time (Min): {int(np.floor((time.time() - start)/60))} ' + 
              f'lr: { lr:.4f}')
        net.train()
        net_t.eval()
        
        
        # Define the sampler for each epoch
        sampler = SubsetRandomSampler(indices[:60000]) 
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        for xs, ys in train_loader:
            xs, ys = Variable(xs), Variable(ys)
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()            
            
            #attack = torchattacks.AutoAttack(net,norm='Linf', eps=EPS, n_classes=10)
            
            '''
            CUT-MIX
            '''
            r = np.random.rand(1)
            if beta > 0 and r < cutmix_prob:
                #print('=> cutmix')
                # generate mixed sample
                lam = np.random.beta(beta, beta)
                rand_index = torch.randperm(xs.size()[0]).cuda()
                target_a = ys
                target_b = ys[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(xs.size(), lam)
                xs[:, :, bbx1:bbx2, bby1:bby2] = xs[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (xs.size()[-1] * xs.size()[-2]))
                # compute output
                adv_a = attack(xs, target_a)
                adv_b = attack(xs, target_b)
                preds_t = net_t(xs)
                preds = net(xs)
                preds_a =  net(adv_a)
                preds_b =  net(adv_b) 
                loss = lossCalculator(benign=preds, targets=target_a, adversarial=preds_a, teacher=preds_t) * lam + lossCalculator(benign=preds, targets=target_b, adversarial=preds_b, teacher=preds_t) * (1. - lam)
                preds_np_a = preds_a.cpu().detach().numpy()
                preds_np_b = preds_b.cpu().detach().numpy()
                correct += (np.argmax(preds_np_a, axis=1) ==
                            target_a.cpu().detach().numpy()).sum()
                correct += (np.argmax(preds_np_b, axis=1) ==
                            target_b.cpu().detach().numpy()).sum()
            else:
                # compute output
                #adv = attack(xs, ys)
                adv = APGD(model=net,x_natural= xs, teacher=net_t, loss=training_loss, T=temperature, alpha=distillation_weight)
                preds_t = net_t(xs)
                preds =  net(xs)
                preds_s =  net(adv)
                #loss = CELoss(xs, ys)
                loss = lossCalculator(benign=preds, targets=ys, adversarial=preds_s, teacher=preds_t)
                _, predicted = torch.max(preds_s, 1)
                correct += (predicted == ys).sum().item()
                #printLogits(outputs= preds, teacher=preds_t, targets= ys, index=0, message= 'preds', escape=False)
            train_losses.append(loss.data.item()) # record training loss
            
            total += xs.size(0)

            step += 1
            optimizer.zero_grad()
            loss.backward()  # calc gradients
            optimizer.step()  # update gradients
            #if total % 1000 == 0:
        acc = float(correct) / total
        print('[%s] Adv KD Training accuracy: %.2f%%' %
                (step, acc * 100))
        total = 0
        correct = 0
        log = lossCalculator.get_log()
        valid_loss = 0.0
        val_acc = 0.0
        if _epoch >= EVAL_INDEX:
            valid_losses, val_acc = evalAdvAttack(net, val_loader)
            valid_loss = np.average(valid_losses)
            avg_valid_losses.append(valid_loss)

        #scheduler.step()
        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)
        epoch_len = len(str(nb_epochs))
        
        print_msg = (f'[{_epoch + 1:>{epoch_len}}/{nb_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' +
                     f'Train Acc: {acc:.5f} ' +
                     f'Val Acc: {val_acc:.5f}')
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'val_acc': val_acc,
            'network':network,
            'classes':classes,
            'epoch': _epoch + 1
                    }
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(_epoch+1, val_acc, net, VERSION + '_' + training_loss + '_A_' + str(distillation_weight) + '_T_' + str(temperature) , state)
        with open(logname, 'a', newline='') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([f'{_epoch + 1}', f'{train_loss:.6f}', f'{valid_loss:.6f}', f'{acc:.3f}', f'{val_acc:.3f}', f'{lr:.4f}', training_loss, log, distillation_weight, temperature, int(early_stopping.counter), f'{int(np.floor((time.time() - start)/60))}' ])
            
        if early_stopping.early_stop:
            print("Early stopping")
            break

def APGD(model, x_natural, teacher, loss, T=30.0, alpha =0.9):
    '''
    EPS = 8/255
    ALPHA = 2/255
    STEPS = 10
    '''
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    MSELoss = nn.MSELoss() 
    step_size= 2/255  # 0.003
    epsilon= 8/255    # 0.031
    perturb_steps=10  # 10
        
    model.eval()
    # generate adversarial example
    delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
    #delta = torch.empty_like(x_natural).uniform_(-epsilon, epsilon).cuda().detach()
    #x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    x_adv = x_natural.detach() + delta
    x_adv2 = x_natural.detach() - delta
    b_logits_T = teacher(x_natural)
    b_logits_S = model(x_natural)
    for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                if loss == 'kl_2':
                    edge2 = criterion_kl(F.log_softmax(model(x_adv)/T, dim=1), F.softmax(b_logits_T/T, dim=1))
                    loss_kl = edge2
                elif loss == 'kl_1_2':
                    edge1 = criterion_kl(F.log_softmax(b_logits_S/T, dim=1), F.softmax(b_logits_T/T, dim=1)) 
                    edge2 = criterion_kl(F.log_softmax(model(x_adv)/T, dim=1), F.softmax(b_logits_T/T, dim=1)) 
                    loss_kl = .5 * (edge1 + edge2)
                elif loss == 'kl_1_3':
                    edge1 = criterion_kl(F.log_softmax(b_logits_S/T, dim=1), F.softmax(b_logits_T/T, dim=1))
                    edge3 = criterion_kl(F.log_softmax(model(x_adv)/T, dim=1), F.softmax(b_logits_S/T, dim=1))
                    loss_kl = .5 * (edge1 + edge3)
                elif loss == 'kl_2_3':
                    edge2 = criterion_kl(F.log_softmax(model(x_adv)/T, dim=1), F.softmax(b_logits_T/T, dim=1))   
                    edge3 = criterion_kl(F.log_softmax(model(x_adv)/T, dim=1), F.softmax(b_logits_S/T, dim=1)) 
                    loss_kl = (1-alpha) * edge2 + alpha * edge3
                elif loss == 'kl_2_3b' or loss == 'kl_2_3a':   # introducing the variant with kl_2_3 loss
                    #edge2v = criterion_kl(F.log_softmax(model(x_adv2)/T, dim=1), F.softmax(b_logits_T/T, dim=1))   
                    edge3v = criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(b_logits_S, dim=1))
                    loss_kl =  edge3v
                elif loss == 'kl_1_2_3':
                    edge1 = criterion_kl(F.log_softmax(b_logits_S/T, dim=1), F.softmax(b_logits_T/T, dim=1)) 
                    edge2 = criterion_kl(F.log_softmax(model(x_adv)/T, dim=1), F.softmax(b_logits_T/T, dim=1))   
                    edge3 = criterion_kl(F.log_softmax(model(x_adv)/T, dim=1), F.softmax(b_logits_S/T, dim=1)) 
                    loss_kl =  (edge1 + edge2 + edge3) / 3
                elif loss == 'trades':
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(model(x_natural), dim=1)) 
                    
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    return x_adv

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# Evaluate results on clean data
def evalClean(net=None, val_loader=None):
    print("Evaluating model results on clean data")
    total = 0
    correct = 0
    net.eval()
    criterion = nn.CrossEntropyLoss()
    # to track the validation loss as the model trains
    valid_losses = []
    with torch.no_grad():
        for xs, ys in val_loader:
            xs, ys = Variable(xs), Variable(ys)
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()
            outputs = net(xs)
            loss = criterion(outputs, ys)
            valid_losses.append(loss.data.item())
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == ys).sum().item()
            total += xs.size(0)
    acc = float(correct) / total
    #print('Clean accuracy: %.2f%%' % (acc * 100))
    return valid_losses, acc

# Evaluate results on adversarially perturbed
def evalAdvAttack(net=None, val_loader=None):
    print("Evaluating model results on adv data")
    total = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    # to track the validation loss as the model trains
    valid_losses = []
    attack = torchattacks.PGD(net, eps=EPS, alpha=ALPHA, steps=STEPS*2)
    net.eval()
    for xs, ys in val_loader:
        if torch.cuda.is_available():
            xs, ys = xs.cuda(), ys.cuda()
        # pytorch PGD
        
        xs, ys = Variable(xs), Variable(ys)
        #adv = attack(xs, ys).detach().cpu()
        adv = attack(xs, ys)
        outputs = net(adv)
        loss = criterion(outputs, ys)
        valid_losses.append(loss.data.item())
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == ys).sum().item()
        total += xs.size(0)
    acc = float(correct) / total
    #print('Adv accuracy: {:.3f}%'.format(acc * 100))
    return valid_losses, acc

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if args.classes == 10:
        train_dataset, val_dataset, test_dataset, train_size = dataset_npz.get_loader(
            args.val_size, args.batch_size)
    elif args.classes == 100:
        train_dataset, val_dataset, test_dataset, train_size = dataset_npz_100.get_loader(
            args.val_size, args.batch_size)
    else:
        raise AssertionError(
            "please choose classes (--classes) 10 for cifar10, 100 for cifar100")

    
    print('==> Preparing Log-File')
    if not os.path.isdir('results'):
        os.mkdir('results')
    logname = ('./results/log_' + args.method + '_' + args.version  + '.csv')
    if not os.path.exists(logname):
        with open(logname, 'a', newline='') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            if(args.method == STANDARD or args.method == ADVERSARIAL):
                logwriter.writerow(['_epoch', 'train_loss', 'valid_loss', 'acc', 'val_acc', 'lr', 'type', 'early_stopping', 'elapsed_time'])
            else:
                logwriter.writerow(['_epoch', 'train_loss', 'valid_loss', 'acc', 'val_acc', 'lr', 'type', 'hard_loss', 'soft_loss', 'total_loss', 'alpha', 'temp', 'early_stopping', 'elapsed_time'])
   
    # Model
    print('==> Building model..')
    print('==> network:', args.network)
    net, DECAY = get_model_by_name(args.network, num_classes= args.classes)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=DEVICES_IDS)
        cudnn.benchmark = True
        
    if args.method == STANDARD:
        Train(logname, net, DECAY, args.network, args.classes, args.batch_size, train_size, train_dataset, val_dataset, args.beta, args.cutmix_prob, args.epochs, args.lr, args.patience, args.version)
    elif args.method == ADVERSARIAL:
        advTrain(logname, net, DECAY, args.network, args.classes, args.batch_size, train_size, train_dataset, val_dataset, args.beta, args.cutmix_prob, args.epochs, args.lr, args.patience, args.version)
    elif args.method == ALPISTILLATION:
        advALPTrain(logname, net, DECAY, args.network, args.classes, args.batch_size, train_size, device, train_dataset, val_dataset, args.beta, args.cutmix_prob, args.epochs,args.distillation_weight, args.temperature,
                  args.loss, args.lr, args.patience, args.version)
    elif args.method == KDISTILLATION:
        net_t, DECAY = get_model_by_name(args.network, num_classes= args.classes)
        net_t = net_t.to(device)
        if device == 'cuda':
            net_t = torch.nn.DataParallel(net_t, device_ids=DEVICES_IDS)
            cudnn.benchmark = True

        # load teacher
        load_path = "./checkpoint/"
        checkpoint = torch.load(load_path + args.teacher + '_.pth',
                                map_location=lambda storage, loc: storage.cuda(0))['net']
        net_t.load_state_dict(checkpoint)
        net_t.eval() # auto ignore dropout layer
        #print(net_t)
        print('==> loaded Teacher')
        advKDTrain(logname, net, DECAY, args.network, args.classes, args.batch_size, train_size, net_t, device, train_dataset, val_dataset, args.beta, args.cutmix_prob, args.epochs,args.distillation_weight, args.temperature,
                  args.loss, args.lr, args.patience, args.version)
    else:
        raise AssertionError(
            "please choose trainnig (--type) std:Standard Train, adv: Adversarial Training, kd: Adversarial Knowledge Distillation")

    #evalClean(net, val_dataset)
    #evalAdvAttack(net, val_dataset)


def adjust_learning_rate(learning_rate,optimizer, epoch):
    lr = learning_rate
    if epoch >= 75:
        lr /= 10
    if epoch >= 90:
        lr /= 10
    if epoch >= 100:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer, lr

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument('--network', type=str, default="resnet18")
    parser.add_argument('--classes', type=int, default=10)
    parser.add_argument('--method', type=str, default="nt", help='method used (nt:Standard, at:adversarial, alp:logit pairing, kd:knowledge distillation)')
    parser.add_argument('--loss', type=str, default="kd")
    parser.add_argument('--beta', default=0.0, type=float,
                    help='hyperparameter beta - 0.0 CutMix is not used, 1.0 CutMix used')
    parser.add_argument('--cutmix_prob', default=0.5, type=float,
                    help='cutmix probability')
    parser.add_argument('--val_size', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--teacher', type=str, default="NT")
    parser.add_argument('--version', type=str, default="v1")
    parser.add_argument('--temperature', default=1.0,
                        type=float, help='KD Loss Temperature')
    parser.add_argument('--distillation_weight', default=0.5,
                        type=float, help=' KD distillation weight / ALPHA: 0-1')
    parser.add_argument('--lr', default=0.1,
                        type=float, help='learning rate')
    parser.add_argument('--dropout', '-d', action='store_true',
                        help='adds dropout after each relu in the network of P=0.7')
    parser.add_argument('--drate', default=0.7,
                        type=float, help='dropout rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()
    main(args)