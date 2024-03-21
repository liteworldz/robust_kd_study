import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm

#from tvision import models

from resnet import *
from wideresnet import WideResNet

#import preactresnet
#from models import *
import dataset_npz
import dataset_npz_100

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import torchattacks
import argparse
import numpy as np
from utils import progress_bar
import time
import os
import csv

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'

EPS= 8/255
ALPHA= 2/255
STEPS= 20

# Evaluate results on clean data
def evalClean(model, dataloader, criterion, device='cuda'):
    print("Evaluating model results on clean data")
    """
    Evaluate a PyTorch model on clean data.

    Args:
    - model: PyTorch model to evaluate
    - dataloader: PyTorch DataLoader for the clean data
    - criterion: Loss function to evaluate the model
    - device: Device to perform the evaluation on (default: 'cuda')

    Returns:
    - loss: Average loss on the clean data
    - accuracy: Accuracy on the clean data
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in tqdm(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == targets).sum().item()
        total_samples += inputs.size(0)

    # Calculate average loss and accuracy
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    #print('accuracy: {:.3f}％'.format(accuracy * 100))
    return avg_loss, accuracy
    

# Evaluate results on adversarially perturbed
def evalAdvAttack(model, dataloader, criterion, attack, device='cuda'):
    print("Evaluating model results on adv data")
    """
    Evaluate a PyTorch model on adv data.

    Args:
    - model: PyTorch model to evaluate
    - dataloader: PyTorch DataLoader for the clean data
    - criterion: Loss function to evaluate the model
    - attack: attack to evaluate
    - device: Device to perform the evaluation on (default: 'cuda')

    Returns:
    - loss: Average loss on the adv data
    - accuracy: Accuracy on the adv data
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in tqdm(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        adv = attack(inputs, targets)
        outputs = model(adv)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == targets).sum().item()
        total_samples += inputs.size(0)

    # Calculate average loss and accuracy
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    #print('accuracy: {:.3f}％'.format(accuracy * 100))
    return avg_loss, accuracy

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

    if classes == 10:
        train_dataset, val_dataset, test_dataset, train_size = dataset_npz.get_loader(
            val_size, batch_size)
    elif classes == 100:
        train_dataset, val_dataset, test_dataset, train_size = dataset_npz_100.get_loader(
            val_size, batch_size)
        
    # Define the indices for the entire dataset
    indices = list(range(train_size))

    # Define the sampler for each epoch
    sampler = SubsetRandomSampler(indices[:60000]) 
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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
    valid_losses_clean, acc_clean = evalClean(net, data_loader, criterion=criterion)
    valid_losses_adv, acc_adv = evalAdvAttack(net, data_loader, criterion=criterion, attack=attack)
    
    print('\nTotal benign [',data_type,'] accuarcy:', 100*acc_clean)
    print('Total adversarial [',data_type,'] Accuarcy:',100* acc_adv)
    #print('Total benign test loss:', benign_loss)
    #print('Total adversarial test loss:', adv_loss)acc
    print("Elapsed Time (Min): ", np.floor((time.time() - start)/60))
    
    return trainEpochs, 100*acc_clean, 100* acc_adv


def init_test(args,device):
    
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
    parser.add_argument('--val_size', type=int, default=1000)
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