import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import defaultdict
import sys

class ProjectionModule(nn.Module):
    def __init__(self):
        super(ProjectionModule, self).__init__()
        
    def forward(self, x1, x2):
        # Multiply each tensor with its transpose to get square matrix of each
        x1_square = torch.matmul(x1, x1.t())
        x2_square = torch.matmul(x2, x2.t())
        
        # Compute eigendecomposition of each matrix
        e1, v1 = torch.linalg.eig(x1_square)
        e2, v2 = torch.linalg.eig(x2_square)
        
        # Sort eigenvectors by eigenvalues
        _, indices1 = torch.sort(torch.abs(e1[:, 0]), descending=True)
        _, indices2 = torch.sort(torch.abs(e2[:, 0]), descending=True)
        
        # Get the top 10 eigenvectors for each matrix
        top_v1 = v1[:, indices1[:10]]
        top_v2 = v2[:, indices2[:10]]
        
        # Add the two resulted projections
        projection = torch.matmul(top_v1, top_v1.t()) + torch.matmul(top_v2, top_v2.t())
        
        # Reconstruct to size [200,10]
        reconstruction = torch.matmul(projection, x1)
        
        return reconstruction

class MatrixCompressionModule(nn.Module):
    def __init__(self):
        super(MatrixCompressionModule, self).__init__()
        self.linear = nn.Linear(200, 10)
    
    def forward(self, x1, x2):
        # Get the size of the tensors
        batch_size = x1.size(0)
        num_features = x1.size(1)
        
        # Compute the square matrices for x1 and x2
        x1_square = torch.matmul(x1, x1.t())
        x2_square = torch.matmul(x2, x2.t())
        
        # Apply compression to each matrix
        x1_compressed = self.linear(x1_square.view(batch_size, -1))
        x2_compressed = self.linear(x2_square.view(batch_size, -1))
        
        # Add the two matrices and reshape back to the original size
        result = (x1_compressed + x2_compressed).view(batch_size, num_features)
        
        return result
    
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
        self.ProjLogits          = MatrixCompressionModule()

    def forward(self, benign, targets, adversarial):
        # Distillation Loss
        soft_target_loss = 0.0
        hard_target_loss = 0.0
        
        projLogits = .5 * (self.ProjLogits(benign, adversarial) + adversarial)
        
        if self.training_loss == 'alp':
            soft_target_loss = self.MSELoss(adversarial,  benign)
            hard_target_loss = .5 * (F.cross_entropy(benign, targets, reduction='mean') + F.cross_entropy(adversarial, targets, reduction='mean'))
            total_loss =  hard_target_loss  + (soft_target_loss  * self.distillation_weight)
        elif self.training_loss == 'alpkdce1':
            soft_target_loss = self.kldiv(F.log_softmax(adversarial/self.temperature, dim=1), 
                                        F.softmax(benign/self.temperature, dim=1)) * (self.temperature ** 2)
            hard_target_loss = .5 * (F.cross_entropy(benign, targets, reduction='mean') + F.cross_entropy(adversarial, targets, reduction='mean'))
            total_loss = ( hard_target_loss *  (1 -  self.distillation_weight)) + (soft_target_loss  * self.distillation_weight)
        elif self.training_loss == 'proj':
            soft_target_loss = self.kldiv(F.log_softmax(adversarial/self.temperature, dim=1), 
                                        F.softmax(projLogits/self.temperature, dim=1)) * (self.temperature ** 2)
            hard_target_loss = F.cross_entropy(projLogits, targets, reduction='mean') 
            total_loss = hard_target_loss  
        
            
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