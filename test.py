import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm

#from tvision import models

from resnet import *
#import preactresnet
#from models import *
import dataset

import torchattacks
import argparse
import numpy as np
from utils import progress_bar
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    #print('Adv accuracy: {:.3f}???????'.format(acc * 100))
    return valid_losses, acc

def test(net, adversary, steps, filename, val_size, batch_size, data_type):
    
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
        net, norm='Linf', eps=EPS, version='standard', n_classes=10, seed=None, verbose=False)
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
    


def init_test(args):
    
    filename = args.filename
    net = ResNet18() #preactresnet.PreActResNet18() #models.resnet18() #
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    
    print('\n==> Using Training Dataset..')
    test(net, 'PGD', 10, filename, args.val_size, args.batch_size, 'train')
    print('\n==> Using Validation Dataset..')
    test(net, 'PGD', 20, filename, args.val_size, args.batch_size, 'val')
    print('\n==> Using Testing Dataset..')
    test(net, 'PGD', 200, filename, args.val_size, args.batch_size, 'test')
    print('\n==> Using Testing Dataset.. (AUTOATTACK)')
    test(net, 'auto', 10, filename, args.val_size, args.batch_size, 'test')
        


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--val_size', type=int, default=6000)
    parser.add_argument('--batch_size', type=int, default=200)
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

    init_test(args)


if __name__ == '__main__':
    main()