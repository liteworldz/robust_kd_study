import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm

#from tvision import models

from resnet import *
from wideresnet import WideResNet
#import preactresnet
#from models import *
import dataset
import dataset_cifar100

import torchattacks
import argparse
import numpy as np
from utils import progress_bar
import time
import os
import csv


EPS= 8/255
ALPHA= 2/255
STEPS= 20

# Evaluate results on clean data
def evalClean(net=None, data_loader=None):
    print("Evaluating model results on clean data")
    total = 0
    correct = 0
    net.eval()
    criterion = nn.CrossEntropyLoss()
    # to track the validation loss as the model trains
    valid_losses = []
    with torch.no_grad():
        for xs, ys in tqdm(data_loader):
            xs, ys = Variable(xs), Variable(ys)
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()
            preds1 = net(xs)
            loss = criterion(preds1, ys)
            valid_losses.append(loss.data.item())
            preds_np1 = preds1.cpu().detach().numpy()
            finalPred = np.argmax(preds_np1, axis=1)
            correct += (finalPred == ys.cpu().detach().numpy()).sum()
            total += len(xs)
    acc = float(correct) / total
    #print('Clean accuracy: %.2f%%' % (acc * 100))
    return valid_losses, acc

# Evaluate results on adversarially perturbed
def evalAdvAttack(net=None, data_loader=None, attack=None):
    print("Evaluating model results on adv data")
    total = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    # to track the validation loss as the model trains
    valid_losses = []
    net.eval()
    for xs, ys in tqdm(data_loader):
        if torch.cuda.is_available():
            xs, ys = xs.cuda(), ys.cuda()
        # pytorch PGD
        
        xs, ys = Variable(xs), Variable(ys)
        #adv = attack(xs, ys).detach().cpu()
        adv = attack(xs, ys)
        preds = net(adv)
        loss = criterion(preds, ys)
        valid_losses.append(loss.data.item())
        preds_np = preds.cpu().detach().numpy()
        finalPred = np.argmax(preds_np, axis=1)
        correct += (finalPred == ys.cpu().detach().numpy()).sum()
        total += data_loader.batch_size
    acc = float(correct) / total
    #print('Adv accuracy: {:.3f}％'.format(acc * 100))
    return valid_losses, acc

def test(net, classes, adversary, steps, filename, val_size, batch_size, data_type):
    
    load_path = "./checkpoint/"
    checkpoint = torch.load(load_path + filename,
                            map_location=lambda storage, loc: storage.cuda(0))['net']
    trainAccuracy = torch.load(load_path + filename,
                               map_location=lambda storage, loc: storage.cuda(0))['acc']
    valAccuracy = torch.load(load_path + filename,
                               map_location=lambda storage, loc: storage.cuda(0))['val_acc']
    trainEpochs = torch.load(load_path + filename,
                             map_location=lambda storage, loc: storage.cuda(0))['epoch']

    net.load_state_dict(checkpoint)

    if classes == 100:
        trainloader, valloader, testloader = dataset_cifar100.get_loader(val_size, batch_size)
    else:
        trainloader, valloader, testloader = dataset.get_loader(val_size, batch_size)

    if data_type=='test':
        data_loader = testloader
    elif data_type=='val':
        data_loader = valloader
    else:
        data_loader = trainloader
    print('==> Loaded Model data..')
    print("Train Acc", trainAccuracy)
    print("Val Acc", valAccuracy)
    print("Best Train Epoch", trainEpochs)
    # Data
    print('==> Preparing data..')
    criterion = nn.CrossEntropyLoss()
    print('\n[ Test Start on ', data_type, ' Dataset]')
    start = time.time()
    loss = 0
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0

    AUTOadversary = torchattacks.AutoAttack(
        net, norm='Linf', eps=EPS, version='standard', n_classes=classes, seed=None, verbose=False)
    PGDadversary = torchattacks.PGD(net, eps=EPS, alpha=ALPHA, steps=steps)
    
    net.eval()
    if adversary =='PGD':
        attack = PGDadversary
    else:
        attack = AUTOadversary
    valid_losses_clean, acc_clean = evalClean(net, data_loader)
    valid_losses_adv, acc_adv = evalAdvAttack(net, data_loader, attack)
    
    print('\nTotal benign [',data_type,'] accuarcy:', 100*acc_clean)
    print('Total adversarial [',data_type,'] Accuarcy:',100* acc_adv)
    #print('Total benign test loss:', benign_loss)
    #print('Total adversarial test loss:', adv_loss)acc
    print("Elapsed Time (Min): ", np.floor((time.time() - start)/60))
    
    return trainEpochs, 100*acc_clean, 100* acc_adv


def init_test(args, device):
    
    filename = args.filename
    if args.network== 'wideresnet':
        net =  WideResNet(28, num_classes=args.classes, widen_factor=10, dropRate=0.0)
    else:
        net = ResNet18(num_classes=args.classes)
        
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True        

    print('\n==> Using Training Dataset..')
    best, train_acc, train_PGD10_acc = test(net, args.classes, 'PGD', 10, filename, args.val_size, args.batch_size, 'train')
    print('\n==> Using Validation Dataset..')
    best, val_acc, val_PGD20_acc = test(net, args.classes, 'PGD', 20, filename, args.val_size, args.batch_size, 'val')
    print('\n==> Using Testing Dataset..')
    best, test_acc, test_PGD200_acc = test(net, args.classes, 'PGD', 200, filename, args.val_size, args.batch_size, 'test')
    print('\n==> Using Testing Dataset.. (AUTOATTACK)')
    best, test_acc, test_auto_acc = test(net, args.classes, 'auto', 10, filename, args.val_size, args.batch_size, 'test')
        
    print('==> Preparing Log-File')
    if not os.path.isdir('results'):
        os.mkdir('results')
    logname = ('./results/log_' + str(args.network) + '_' + str(args.classes)  + '.csv')
    if not os.path.exists(logname):
        with open(logname, 'a', newline='') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['filename','_epoch', 'train_acc', 'train_PGD10_acc','val_acc', 'val_PGD20_acc','test_acc','test_PGD200_acc','test_Auto_acc'])
    with open(logname, 'a', newline='') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([filename, int(best) ,f'{train_acc:.6f}', f'{train_PGD10_acc:.6f}', f'{val_acc:.6f}', f'{val_PGD20_acc:.6f}'
                            , f'{test_acc:.6f}', f'{test_PGD200_acc:.6f}', f'{test_auto_acc:.6f}'])
    


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument('--network', type=str, default="resnet18")
    parser.add_argument('--classes', type=int, default=10)
    parser.add_argument('--val_size', type=int, default=6000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument(
        '--filename',
        default='teacher.pth',
        type=str,
        help='name of the model to test')
    parser.add_argument(
        '--attack',
        default='PGD',
        type=str,
        help='name of the attack')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    init_test(args, device)


if __name__ == '__main__':
    main()