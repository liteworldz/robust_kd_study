import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import defaultdict
import sys

'''
def calculate_loss(logits1, logits2, targets, teacher, alpha=.1):
    # Compute softmax probabilities for logits1 and logits2
    probs1 = F.softmax(logits1, dim=1)
    probs2 = F.softmax(logits2, dim=1) 
    
    # Compute the best probabilities from logits2 for each example
    best_probs2, _ = torch.max(probs2, dim=1)

    # Compute the cross-entropy loss for logits1 with targets
    loss1 = F.cross_entropy(logits2, targets, reduction="mean")
    
    # Compute the cross-entropy loss for logits1 with the best probabilities from logits2
    loss1_best = -torch.log(torch.sum(probs1 * best_probs2[:, None], dim=1)).mean()

    avg_loss1 = torch.mean(loss1)
    avg_loss1_best = torch.mean(loss1_best)
    
    # Compute the weighted average of the losses
    loss = (1 - alpha)  * avg_loss1 + alpha * avg_loss1_best

    return loss
'''

def calculate_loss(logits1, logits2, targets, teacher,teacher_adv, temperature=1.0, alpha=0.9):
    # Compute softmax probabilities for student and teacher logits
    student_probs = F.softmax(teacher/temperature, dim=1)
    teacher_probs = F.softmax(teacher_adv/temperature, dim=1)
    kldiv = nn.KLDivLoss(reduction='batchmean') 
    # Compute the average probabilities between student and teacher logits
    avg_probs = 0.5 * (student_probs + teacher_probs)

    # Compute the JSD loss using KL divergence
    jsd_loss = alpha * kldiv(F.log_softmax(logits2/temperature, dim=1), avg_probs) + (1 - alpha) * kldiv(F.log_softmax(logits1/temperature, dim=1),avg_probs) 
                            
    # Compute the cross-entropy loss with label smoothing
    ce_loss = F.cross_entropy(logits2, targets, reduction='mean')

    # Compute the weighted average of JSD loss and cross-entropy loss
    loss = alpha * jsd_loss + (1 - alpha) * ce_loss

    return jsd_loss

class LossCalulcator(nn.Module):
    def __init__(self, temperature, distillation_weight, epoch, training_loss):
        super().__init__()

        self.temperature         = temperature
        self.distillation_weight = distillation_weight
        self.epoch               = epoch
        self.training_loss       = training_loss
        self.loss_log            = defaultdict(list)
        self.kldiv               = nn.KLDivLoss(reduction='batchmean') 
        self.CELoss              = nn.CrossEntropyLoss()
        self.MSELoss             = nn.MSELoss() 

    def forward(self, benign, targets, adversarial, teacher, teacher_adv=None):
        # Distillation Loss
        soft_target_loss = 0.0
        hard_target_loss = 0.0
        
        return calculate_loss(logits1=benign, logits2= adversarial, targets=targets, teacher=teacher, teacher_adv=teacher_adv)
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
        elif self.training_loss == 'nkd':
            soft_target_loss = self.kldiv(F.log_softmax(benign/self.temperature, dim=1), 
                                        F.softmax(adversarial/self.temperature, dim=1)) * (self.temperature ** 2)
            hard_target_loss = F.cross_entropy(benign, targets, reduction='mean')
            total_loss = ( hard_target_loss *  (1 -  self.distillation_weight)) + (soft_target_loss  * self.distillation_weight)
        
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