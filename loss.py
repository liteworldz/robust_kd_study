import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import defaultdict 
import sys

class LossCalulcator(nn.Module):
    def __init__(self, temperature, distillation_weight, epoch, training_loss):
        super().__init__()

        self.temperature         = temperature
        self.distillation_weight = distillation_weight
        self.epoch               = epoch
        self.training_loss       = training_loss
        self.loss_log            = defaultdict(list)
        self.kldiv               = nn.KLDivLoss(reduction="batchmean") 
        self.CELoss              = nn.CrossEntropyLoss()
        self.MSELoss             = nn.MSELoss() 

    def forward(self, benign, targets, adversarial, teacher):
        # Distillation Loss
        soft_target_loss = 0.0
        hard_target_loss = 0.0 
        
        edge1 = self.kldiv(F.log_softmax(benign/self.temperature, dim=1), F.softmax(teacher/self.temperature, dim=1))  * (self.temperature ** 2)
        edge2 = self.kldiv(F.log_softmax(adversarial/self.temperature, dim=1), F.softmax(teacher/self.temperature, dim=1)) * (self.temperature ** 2)
        edge3 = self.kldiv(F.log_softmax(adversarial/self.temperature, dim=1), F.softmax(benign/self.temperature, dim=1))  * (self.temperature ** 2)
            
            
        if self.training_loss == 'alp':
            soft_target_loss = self.MSELoss(adversarial,  benign)
            hard_target_loss = .5 * (F.cross_entropy(benign, targets, reduction='mean') + F.cross_entropy(adversarial, targets, reduction='mean'))
            total_loss =  hard_target_loss  + (soft_target_loss  * self.distillation_weight)
        elif self.training_loss == 'alpkdce1':
            soft_target_loss = self.kldiv(F.log_softmax(adversarial/self.temperature, dim=1), 
                                        F.softmax(teacher/self.temperature, dim=1)) * (self.temperature ** 2)
            hard_target_loss = .5 * (F.cross_entropy(teacher, targets, reduction='mean') + F.cross_entropy(adversarial, targets, reduction='mean'))
            total_loss = ( hard_target_loss *  (1 -  self.distillation_weight)) + (soft_target_loss  * self.distillation_weight)
        elif self.training_loss == 'alpkdce':
            soft_target_loss = self.kldiv(F.log_softmax(adversarial/self.temperature, dim=1), 
                                        F.softmax(teacher/self.temperature, dim=1)) * (self.temperature ** 2)
            hard_target_loss = .5 * (F.cross_entropy(benign, targets, reduction='mean') + F.cross_entropy(adversarial, targets, reduction='mean'))
            total_loss = ( hard_target_loss *  (1 -  self.distillation_weight)) + (soft_target_loss  * self.distillation_weight)
        elif self.training_loss == 'kl_2':
            total_loss = edge2
        elif self.training_loss == 'kl_1_2':
            total_loss = .5 * (edge1  + edge2)
        elif self.training_loss == 'kl_1_3':
            total_loss = .5 * (edge1  + edge3)
        elif self.training_loss == 'kl_2_3' or self.training_loss == 'kl_2_3v':
            total_loss = (1-self.distillation_weight) * edge2 + self.distillation_weight * edge3
        elif self.training_loss == 'kl_1_2_3':
            total_loss =  (edge1 + edge2 + edge3) / 3
        elif self.training_loss == 'trades':
            hard_target_loss = F.cross_entropy(benign, targets)
            soft_target_loss = self.kldiv(F.log_softmax(adversarial, dim=1), F.softmax(benign, dim=1)) 
            total_loss = hard_target_loss + 5.0 * soft_target_loss
        
        
        # Logging
        #if self.distillation_weight > 0:
        self.loss_log['hard_target_loss'].append(hard_target_loss)

        #if self.distillation_weight < 1:
        self.loss_log['soft_target_loss'].append(soft_target_loss)

        self.loss_log['total_loss'].append(total_loss)

        return total_loss

    def get_log(self, length=100):
        log = []
        # calucate the average value from lastest N losses
        for key in self.loss_log.keys():
            if len(self.loss_log[key]) < length:
                length = len(self.loss_log[key])
            log.append("%2.3f"%(sum(self.loss_log[key][-length:]) / length))
        #print(", ".join(log))
        
        return ", ".join(log)