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

def calculate_loss(logits1, logits2, targets, teacher,teacher_adv, temperature=30.0, alpha=0.9): 
    kldiv = nn.KLDivLoss(reduction='batchmean') 
    # Compute the average probabilities between student and teacher logits
    avg_probs = 0.5 * (logits1 + logits2)

    # Compute the JSD loss using KL divergence
    jsd_loss = kldiv(F.log_softmax(logits2/temperature, dim=1),F.softmax(avg_probs/temperature, dim=1)) * (temperature ** 2)
                            
    # Compute the cross-entropy loss with label smoothing
    ce_loss = F.cross_entropy(logits2, targets, reduction='mean')

    # Compute the weighted average of JSD loss and cross-entropy loss
    loss = alpha * jsd_loss + (1 - alpha) * ce_loss

    return loss



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

    def forward(self, benign, targets, adversarial, teacher):
        # Distillation Loss
        soft_target_loss = 0.0
        hard_target_loss = 0.0
        
        #return calculate_loss(logits1=benign, logits2= adversarial, targets=targets, teacher=teacher, teacher_adv=teacher_adv)
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
        elif self.training_loss == 'rslad':
            soft_target_loss = self.kldiv(F.log_softmax(benign, dim=1), 
                                        F.softmax(teacher, dim=1))  
            hard_target_loss = self.kldiv(F.log_softmax(adversarial, dim=1), 
                                        F.softmax(teacher, dim=1)) 
            total_loss = ( hard_target_loss *  (1 -  self.distillation_weight)) + (soft_target_loss  * self.distillation_weight)
        elif self.training_loss == 'trades':
            soft_target_loss = self.kldiv(F.log_softmax(adversarial, dim=1), 
                                          F.softmax(benign, dim=1))
            hard_target_loss = F.cross_entropy(benign, targets, reduction='mean')
            # The trade-off regularization parameter beta can be set in [1, 10]. Larger beta leads to more robust and less accurate models.
            total_loss = hard_target_loss +  10.0 * soft_target_loss
        elif self.training_loss == 'custom1':
            soft_target_loss = self.kldiv(F.log_softmax(benign/self.temperature, dim=1), 
                                        F.softmax(adversarial/self.temperature, dim=1)) * (self.temperature ** 2) 
            hard_target_loss = self.kldiv(F.log_softmax(adversarial/self.temperature, dim=1), 
                                        F.softmax(teacher/self.temperature, dim=1)) * (self.temperature ** 2)
            total_loss = ( hard_target_loss *  (1 -  self.distillation_weight)) + (soft_target_loss  * self.distillation_weight)
        elif self.training_loss == 'custom3':
            soft_target_loss = self.kldiv(F.log_softmax(benign, dim=1), 
                                        F.softmax(adversarial, dim=1))
            hard_target_loss = self.kldiv(F.log_softmax(adversarial/self.temperature, dim=1), 
                                        F.softmax(teacher/self.temperature, dim=1)) * (self.temperature ** 2)
            total_loss = ( hard_target_loss *  (1 -  self.distillation_weight)) + (soft_target_loss  * self.distillation_weight)
        elif self.training_loss == 'custom5':
            avd_probes = .5*(benign+adversarial)
            soft_target_loss = self.kldiv(F.log_softmax(adversarial, dim=1), 
                                        F.softmax(avd_probes, dim=1))
            hard_target_loss = self.kldiv(F.log_softmax(avd_probes/self.temperature, dim=1), 
                                        F.softmax(teacher/self.temperature, dim=1)) * (self.temperature ** 2)
            total_loss = ( hard_target_loss *  (1 -  self.distillation_weight)) + (soft_target_loss  * self.distillation_weight)
        elif self.training_loss == 'custom6':
            avd_probes1 = .5*(benign+adversarial)
            avd_probes2 = .5*(teacher+adversarial)
            soft_target_loss = self.kldiv(F.log_softmax(adversarial, dim=1), 
                                        F.softmax(avd_probes1, dim=1))
            hard_target_loss = self.kldiv(F.log_softmax(avd_probes2/self.temperature, dim=1), 
                                        F.softmax(teacher/self.temperature, dim=1)) * (self.temperature ** 2)
            total_loss = ( hard_target_loss *  (1 -  self.distillation_weight)) + (soft_target_loss  * self.distillation_weight)
        elif self.training_loss == 'jsdg':
            avd_probes = (benign+adversarial+teacher) / 3
            total_loss = ((self.kldiv(F.log_softmax(benign/self.temperature, dim=1), F.softmax(avd_probes/self.temperature, dim=1)) * (self.temperature ** 2))
                           + (self.kldiv(F.log_softmax(adversarial/self.temperature, dim=1), F.softmax(avd_probes/self.temperature, dim=1)) * (self.temperature ** 2))
                           + (self.kldiv(F.log_softmax(teacher/self.temperature, dim=1), F.softmax(avd_probes/self.temperature, dim=1)) * (self.temperature ** 2))) /3
        elif self.training_loss == 'jsdg2':
            avd_probes = (benign+adversarial+teacher) / 3
            total_loss = (self.kldiv(F.log_softmax(benign, dim=1), F.softmax(avd_probes, dim=1))
                           + self.kldiv(F.log_softmax(adversarial, dim=1), F.softmax(avd_probes, dim=1))
                           + (self.kldiv(F.log_softmax(teacher/self.temperature, dim=1), F.softmax(avd_probes/self.temperature, dim=1)) * (self.temperature ** 2))) /3
        elif self.training_loss == 'jsd1':
            avd_probes = .5*(benign+adversarial)
            soft_target_loss = self.kldiv(F.log_softmax(adversarial, dim=1), 
                                        F.softmax(avd_probes, dim=1))
            hard_target_loss = self.kldiv(F.log_softmax(teacher/self.temperature, dim=1), 
                                        F.softmax(avd_probes/self.temperature, dim=1)) * (self.temperature ** 2)
            total_loss = .5 * ( hard_target_loss  + soft_target_loss )
        elif self.training_loss == 'jsd2':
            avd_probes1 = .5*(benign+adversarial)
            avd_probes2 = .5*(teacher+adversarial)
            soft_target_loss = self.kldiv(F.log_softmax(adversarial, dim=1), 
                                        F.softmax(avd_probes1, dim=1))
            hard_target_loss = self.kldiv(F.log_softmax(teacher/self.temperature, dim=1), 
                                        F.softmax(avd_probes2/self.temperature, dim=1)) * (self.temperature ** 2)
            #total_loss = .5 * ( hard_target_loss  + soft_target_loss )
            total_loss = ( hard_target_loss *  (1 -  self.distillation_weight)) + (soft_target_loss  * self.distillation_weight)
        elif self.training_loss == 'kl_2':
            edge1 = self.kldiv(F.log_softmax(benign/self.temperature, dim=1), F.softmax(teacher/self.temperature, dim=1))  * (self.temperature ** 2)
            edge2 = self.kldiv(F.log_softmax(adversarial/self.temperature, dim=1), F.softmax(teacher/self.temperature, dim=1)) * (self.temperature ** 2)
            edge3 = self.kldiv(F.log_softmax(benign/self.temperature, dim=1), F.softmax(adversarial/self.temperature, dim=1)) * (self.temperature ** 2)
            total_loss = edge2
        elif self.training_loss == 'kl_1_2':
            edge1 = self.kldiv(F.log_softmax(benign/self.temperature, dim=1), F.softmax(teacher/self.temperature, dim=1))  * (self.temperature ** 2)
            edge2 = self.kldiv(F.log_softmax(adversarial/self.temperature, dim=1), F.softmax(teacher/self.temperature, dim=1)) * (self.temperature ** 2)
            edge3 = self.kldiv(F.log_softmax(benign/self.temperature, dim=1), F.softmax(adversarial/self.temperature, dim=1)) * (self.temperature ** 2)
            total_loss = .5 * (edge1  + edge2)
        elif self.training_loss == 'kl_1_3':
            edge1 = self.kldiv(F.log_softmax(benign/self.temperature, dim=1), F.softmax(teacher/self.temperature, dim=1))  * (self.temperature ** 2)
            edge2 = self.kldiv(F.log_softmax(adversarial/self.temperature, dim=1), F.softmax(teacher/self.temperature, dim=1)) * (self.temperature ** 2)
            edge3 = self.kldiv(F.log_softmax(benign/self.temperature, dim=1), F.softmax(adversarial/self.temperature, dim=1)) * (self.temperature ** 2)
            total_loss = .5 * (edge1  + edge3)
        elif self.training_loss == 'kl_2_3':
            edge1 = self.kldiv(F.log_softmax(benign/self.temperature, dim=1), F.softmax(teacher/self.temperature, dim=1)) * (self.temperature ** 2)
            edge2 = self.kldiv(F.log_softmax(adversarial/self.temperature, dim=1), F.softmax(teacher/self.temperature, dim=1)) * (self.temperature ** 2)
            edge3 = self.kldiv(F.log_softmax(benign/self.temperature, dim=1), F.softmax(adversarial/self.temperature, dim=1)) * (self.temperature ** 2)
            total_loss = .5 * (edge2  + edge3)
        elif self.training_loss == 'kl_1_2_3':
            edge1 = self.kldiv(F.log_softmax(benign/self.temperature, dim=1), F.softmax(teacher/self.temperature, dim=1)) * (self.temperature ** 2)
            edge2 = self.kldiv(F.log_softmax(adversarial/self.temperature, dim=1), F.softmax(teacher/self.temperature, dim=1)) * (self.temperature ** 2)
            edge3 = self.kldiv(F.log_softmax(benign/self.temperature, dim=1), F.softmax(adversarial/self.temperature, dim=1)) * (self.temperature ** 2)
            total_loss =  (edge1  + edge2 + edge3) / 3
       
        elif self.training_loss == 'jsd_m':
            # JS(B, C) = 0.5 * (KL(B || M) + KL(C || M)) , where M = 0.5 * (B + C) is the average distribution.
            # Loss = JS(B, C) + KL(B || M)
            M = .5*(adversarial+teacher)
            JS_AB = .5 * ((self.kldiv(F.log_softmax(adversarial/self.temperature, dim=1), F.softmax(M/self.temperature, dim=1))  * (self.temperature ** 2)) + (self.kldiv(F.log_softmax(teacher/self.temperature, dim=1), F.softmax(M/self.temperature, dim=1))  * (self.temperature ** 2)) )
            total_loss = .5*(JS_AB + self.kldiv(F.log_softmax(adversarial/self.temperature, dim=1), F.softmax(M/self.temperature, dim=1)) * (self.temperature ** 2))
        elif self.training_loss == 'jsd_m2':
            # JS(A, C) = 0.5 * (KL(A || M) + KL(C || M)) , where M = 0.5 * (A + C) is the average distribution.
            # Loss = JS(A, C) + KL(B || M)
            M = .5*(benign+teacher)
            JS_AB = .5 * ((self.kldiv(F.log_softmax(benign/self.temperature, dim=1), F.softmax(M/self.temperature, dim=1))  * (self.temperature ** 2)) + (self.kldiv(F.log_softmax(teacher/self.temperature, dim=1), F.softmax(M/self.temperature, dim=1))  * (self.temperature ** 2)) )
            total_loss = .5*(JS_AB + self.kldiv(F.log_softmax(adversarial/self.temperature, dim=1), F.softmax(M/self.temperature, dim=1)) * (self.temperature ** 2))
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