# Source : https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation acc doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='./checkpoint/', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation acc improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation acc improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, epoch, val_acc, model, type, state):

        score = val_acc
        # alwyas save the last epoch
        if epoch == 200:
            self.save_checkpoint(val_acc, model, type, state)
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model, type, state)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} ')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model, type, state)
            self.counter = 0
        
        
    def save_checkpoint(self, val_acc, model, type, state):
        '''Saves model when validation acc increased.'''
        if self.verbose:
            self.trace_func(f'Validation acc increased ({self.val_acc_min:.6f} --> {val_acc:.6f}).  Saving model ...')
        if not os.path.isdir(self.path + type):
            os.mkdir(self.path + type)
        torch.save(state, self.path + type + '/' + f'_{state["epoch"] }_' + type  + '_.pth')
        self.val_acc_min = val_acc