'''Dual Adversaries'''
adv1, adv2 = evalAPGD(model=net, x_natural=xs, teacher=net_t, loss=training_loss, val_loader=val_loader)
adv = torch.cat([adv1, adv2], dim=0)
xs = torch.cat([xs, xs], dim=0)
ys = torch.cat([ys, ys], dim=0)

def evalAPGD(model, x_natural, teacher, loss, val_loader=None, T=30.0, alpha=0.9):
    # Constants for perturbation
    EPS = 8/255
    ALPHA = 2/255
    STEPS = 10
    
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    step_size = ALPHA
    epsilon = EPS
    perturb_steps = STEPS
    
    model.eval()
    
    # Generate initial adversarial examples
    delta = torch.empty_like(x_natural).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = x_natural.detach() + delta
    x_adv2 = x_natural.detach() - delta
    
    # Get teacher and model logits
    b_logits_T = teacher(x_natural)
    b_logits_S = model(x_natural)
    
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        x_adv2.requires_grad_()
        
        with torch.enable_grad():
            # Compute KL divergence losses
            edge1 = criterion_kl(F.log_softmax(b_logits_S/T, dim=1), F.softmax(b_logits_T/T, dim=1))
            edge2 = criterion_kl(F.log_softmax(model(x_adv2)/T, dim=1), F.softmax(b_logits_T/T, dim=1))
            edge3 = criterion_kl(F.log_softmax(model(x_adv)/T, dim=1), F.softmax(model(x_adv2)/T, dim=1))
            
            # Determine loss based on selected loss type
            if loss == 'kl_2':
                loss_kl = edge2
            elif loss == 'kl_1_2':
                loss_kl = 0.5 * (edge1 + edge2)
            elif loss == 'kl_1_3':
                loss_kl = 0.5 * (edge1 + edge3)
            elif loss == 'kl_2_3':
                loss_kl = (1 - alpha) * edge2 + alpha * edge3
            elif loss == 'kl_1_2_3':
                loss_kl = (edge1 + edge2 + edge3) / 3
            
        grad = torch.autograd.grad(loss_kl, [x_adv], retain_graph=True)[0]
        grad2 = torch.autograd.grad(loss_kl, [x_adv2], retain_graph=True)[0]

        # Update adversarial examples with gradient sign
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv2 = x_adv2.detach() + step_size * torch.sign(grad2.detach())
        
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv2 = torch.min(torch.max(x_adv2, x_natural - epsilon), x_natural + epsilon)
        
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv2 = torch.clamp(x_adv2, 0.0, 1.0)
    
    model.train()
    return x_adv, x_adv2


'''Delta2'''
def APGD(model, x_natural, teacher, loss, T=30.0, alpha =0.9):
    '''
    EPS = 8/255
    ALPHA = 2/255
    STEPS = 10
    '''
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    step_size= 2/255  # 0.003
    epsilon= 8/255    # 0.031
    perturb_steps=10  # 10

    model.eval()
    # generate adversarial example
    delta = torch.empty_like(x_natural).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = x_natural.detach() + delta
    x_adv2 = x_natural.detach() - delta

    b_logits_T = teacher(x_natural)
    b_logits_S = model(x_natural)
    #x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
            x_adv.requires_grad_()
            x_adv2.requires_grad_()
            with torch.enable_grad():
                edge1 = criterion_kl(F.log_softmax(b_logits_S/T, dim=1), F.softmax(b_logits_T/T, dim=1)) 
                edge2 = criterion_kl(F.log_softmax(model(x_adv2)/T, dim=1), F.softmax(b_logits_T/T, dim=1)) 
                edge3 = criterion_kl(F.log_softmax(model(x_adv)/T, dim=1), F.softmax(model(x_adv2)/T, dim=1)) 
                if loss == 'kl_2':
                    loss_kl = edge2
                elif loss == 'kl_1_2':
                    loss_kl = .5 * (edge1 + edge2)
                elif loss == 'kl_1_3':
                    loss_kl = .5 * (edge1 + edge3)
                elif loss == 'kl_2_3':
                    loss_kl = (1-alpha) * edge2 + alpha * edge3
                elif loss == 'kl_1_2_3':
                    loss_kl =  (edge1 + edge2 + edge3) / 3
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            
    
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    return x_adv

'''Enhanced Delta2'''
def APGD(model, x_natural, teacher, T=30.0, alpha=0.9):
    # Constants for perturbation
    EPS = 8/255
    ALPHA = 2/255
    STEPS = 10
    
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    step_size = ALPHA
    epsilon = EPS
    perturb_steps = STEPS
    
    model.eval()
    
    # Generate initial adversarial examples
    delta = torch.empty_like(x_natural).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = x_natural.detach() + delta
    x_adv2 = x_natural.detach() - delta
    
    # Get teacher and model logits
    b_logits_T = teacher(x_natural)
    b_logits_S = model(x_natural)
    
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        x_adv2.requires_grad_()
        
        with torch.enable_grad():
            # Compute KL divergence losses
            edge2 = criterion_kl(F.log_softmax(model(x_adv2)/T, dim=1), F.softmax(b_logits_T/T, dim=1))
            edge3 = criterion_kl(F.log_softmax(model(x_adv)/T, dim=1), F.softmax(model(x_adv2)/T, dim=1))
            loss_kl = (1 - alpha) * edge2 + alpha * edge3
             
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        
        # Update adversarial examples with gradient sign
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    model.train()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)  # Clamp final adversarial examples
    return x_adv




''' Delta2 Updated'''
def APGD(model, x_natural, teacher, loss, T=30.0, alpha=0.9):
    # Constants for perturbation
    EPS = 8/255
    ALPHA = 2/255
    STEPS = 10
    
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    step_size = ALPHA
    epsilon = EPS
    perturb_steps = STEPS
    
    model.eval()
    
    # Generate initial adversarial examples
    delta = torch.empty_like(x_natural).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = x_natural.detach() + delta
    x_adv2 = x_natural.detach() - delta
    
    # Get teacher and model logits
    b_logits_T = teacher(x_natural)
    b_logits_S = model(x_natural)
    
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        x_adv2.requires_grad_()
        
        with torch.enable_grad():
            # Compute KL divergence losses
            edge1 = criterion_kl(F.log_softmax(b_logits_S/T, dim=1), F.softmax(b_logits_T/T, dim=1))
            edge2 = criterion_kl(F.log_softmax(model(x_adv2)/T, dim=1), F.softmax(b_logits_T/T, dim=1))
            edge3 = criterion_kl(F.log_softmax(model(x_adv)/T, dim=1), F.softmax(model(x_adv2)/T, dim=1))
            
            # Determine loss based on selected loss type
            if loss == 'kl_2':
                loss_kl = edge2
            elif loss == 'kl_1_2':
                loss_kl = 0.5 * (edge1 + edge2)
            elif loss == 'kl_1_3':
                loss_kl = 0.5 * (edge1 + edge3)
            elif loss == 'kl_2_3':
                loss_kl = (1 - alpha) * edge2 + alpha * edge3
            elif loss == 'kl_1_2_3':
                loss_kl = (edge1 + edge2 + edge3) / 3
            
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        
        # Update adversarial examples with gradient sign
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    model.train()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)  # Clamp final adversarial examples
    return x_adv


''' Weighted Regulizer'''
def APGD(model, x_natural, teacher, T=30.0, alpha=0.9, regularization_weight=1e-3):
    # Constants for perturbation
    EPS = 8/255
    ALPHA = 2/255
    STEPS = 10
    
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    step_size = ALPHA
    epsilon = EPS
    perturb_steps = STEPS
    
    model.eval()
    
    # Generate initial adversarial examples
    delta = torch.empty_like(x_natural).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = x_natural.detach() + delta
    x_adv2 = x_natural.detach() - delta
    
    # Get teacher and model logits
    b_logits_T = teacher(x_natural)
    b_logits_S = model(x_natural)
    
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        x_adv2.requires_grad_()
        
        with torch.enable_grad():
            # Compute KL divergence losses
            edge2 = criterion_kl(F.log_softmax(model(x_adv2)/T, dim=1), F.softmax(b_logits_T/T, dim=1))
            edge3 = criterion_kl(F.log_softmax(model(x_adv)/T, dim=1), F.softmax(model(x_adv2)/T, dim=1))
            loss_kl = (1 - alpha) * edge2 + alpha * edge3
            
            # Compute regularization loss to encourage naturalness
            reg_loss = regularization_weight * torch.mean(torch.square(x_adv - x_adv2))
            
            total_loss = loss_kl - reg_loss
             
        grad = torch.autograd.grad(total_loss, [x_adv])[0]
        
        # Update adversarial examples with gradient sign
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    model.train()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)  # Clamp final adversarial examples
    return x_adv

'''Delta2 - inverse'''
def APGD(model, x_natural, teacher, loss, T=30.0, alpha=0.9):
    # Constants for perturbation
    EPS = 8/255
    ALPHA = 2/255
    STEPS = 10
    
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    step_size = ALPHA
    epsilon = EPS
    perturb_steps = STEPS
    
    model.eval()
    
    # Generate initial adversarial examples
    delta = torch.empty_like(x_natural).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = x_natural.detach() + delta
    x_adv2 = x_natural.detach() - delta
    
    # Get teacher and model logits
    b_logits_T = teacher(x_natural)
    
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        x_adv2.requires_grad_()
        
        with torch.enable_grad():
            # Compute KL divergence losses
            edge2 = criterion_kl(F.log_softmax(model(x_adv+epsilon)/T, dim=1), F.softmax(b_logits_T/T, dim=1))
            edge3 = criterion_kl(F.log_softmax(model(x_adv)/T, dim=1), F.softmax(model(x_adv+epsilon)/T, dim=1))
            
            loss_kl = (1 - alpha) * edge2 + alpha * edge3
           
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        
        # Update adversarial examples with gradient sign
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    model.train()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)  # Clamp final adversarial examples
    return x_adv








'''KD Three Adversarials'''
def APGD(model, teacher, x_natural, T=30.0, epsilon=8/255, alpha=2/255, num_steps=10):
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    model.eval()

    # Generate initial adversarial examples
    delta2 = torch.empty_like(x_natural).uniform_(-epsilon, epsilon).cuda().detach()
    delta3 = torch.empty_like(x_natural).uniform_(-epsilon, epsilon).cuda().detach()
    delta2_3 = torch.empty_like(x_natural).uniform_(-epsilon, epsilon).cuda().detach()

    # Initialize adversarial examples in opposite directions
    x_adv2 = x_natural.clone().detach() + delta2
    x_adv3 = x_natural.clone().detach() + delta3
    x_adv2_3 = x_natural.clone().detach() + delta2_3
    b_logits_T = teacher(x_natural)
    b_logits_S = model(x_natural)
    for _ in range(num_steps):
        x_adv2.requires_grad_()
        x_adv3.requires_grad_()

        # Compute gradients in the positive direction
        with torch.enable_grad():
         # Compute KL divergence losses             
            edge2 = criterion_kl(F.log_softmax(model(x_adv2)/T, dim=1), F.softmax(b_logits_T/T, dim=1)) 
            edge3 = criterion_kl(F.log_softmax(model(x_adv3)/T, dim=1), F.softmax(b_logits_S/T, dim=1))
            edge =  criterion_kl(F.log_softmax(model(x_adv2_3)/T, dim=1), F.softmax(b_logits_T/T, dim=1))  
            edge2_3  = (1 - alpha) * edge + alpha * criterion_kl(F.log_softmax(model(x_adv2_3)/T, dim=1), F.softmax(b_logits_S/T, dim=1))
            
            loss_kl = edge2 +  edge3 + edge2_3
    
        grad_pos = torch.autograd.grad(loss_kl, [x_adv2, x_adv3, x_adv2_3])[0]
        
        
        # Update adversarial examples
        x_adv2 = x_adv2.detach() + alpha * torch.sign(grad_pos)
        x_adv3 = x_adv3.detach() + alpha * torch.sign(grad_neg)

        # Project adversarial examples within epsilon-ball
        x_adv2 = torch.min(torch.max(x_adv2, x_natural - epsilon), x_natural + epsilon)
        x_adv3 = torch.min(torch.max(x_adv3, x_natural - epsilon), x_natural + epsilon)
        
        x_adv2 = torch.clamp(x_adv2, 0.0, 1.0)
        x_adv3 = torch.clamp(x_adv3, 0.0, 1.0)

    model.train()
    
    return x_adv2, x_adv3







'''KD Two Adversarials'''
def APGD(model, teacher, x_natural, T=30.0, epsilon=8/255, alpha=2/255, num_steps=10):
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    model.eval()

    # Generate initial adversarial examples
    delta2 = torch.empty_like(x_natural).uniform_(-epsilon, epsilon).cuda().detach()
    delta3 = torch.empty_like(x_natural).uniform_(-epsilon, epsilon).cuda().detach()

    # Initialize adversarial examples in opposite directions
    x_adv2 = x_natural.clone().detach() + delta2
    x_adv3 = x_natural.clone().detach() + delta3

    b_logits_T = teacher(x_natural)
    b_logits_S = model(x_natural)
    for _ in range(num_steps):
        x_adv2.requires_grad_()
        x_adv3.requires_grad_()

        # Compute gradients in the positive direction
        with torch.enable_grad():
         # Compute KL divergence losses             
            edge2 = criterion_kl(F.log_softmax(model(x_adv2)/T, dim=1), F.softmax(b_logits_T/T, dim=1)) 
            edge3 = criterion_kl(F.log_softmax(model(x_adv3)/T, dim=1), F.softmax(b_logits_S/T, dim=1))
            
            
            loss_kl = (1 - alpha) * edge2 + alpha * edge3
    
        grad_pos = torch.autograd.grad(loss_kl, [x_adv2, x_adv3])[0]
        
        
        # Update adversarial examples
        x_adv2 = x_adv2.detach() + alpha * torch.sign(grad_pos[0])
        x_adv3 = x_adv3.detach() + alpha * torch.sign(grad_pos[1])

        # Project adversarial examples within epsilon-ball
        x_adv2 = torch.min(torch.max(x_adv2, x_natural - epsilon), x_natural + epsilon)
        x_adv3 = torch.min(torch.max(x_adv3, x_natural - epsilon), x_natural + epsilon)
        
        x_adv2 = torch.clamp(x_adv2, 0.0, 1.0)
        x_adv3 = torch.clamp(x_adv3, 0.0, 1.0)

    model.train()
    
    return x_adv2, x_adv3




'''Delta2 Updated-Shortend  AA 49.86%  '''
def APGD(model, x_natural, teacher, T=30.0, alpha=0.9):
    # Constants for perturbation
    EPS = 8/255
    ALPHA = 2/255
    STEPS = 10
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    step_size = ALPHA
    epsilon = EPS
    perturb_steps = STEPS
    
    model.eval()
    
    # Generate initial adversarial examples
    delta = torch.empty_like(x_natural).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = x_natural.detach() + delta
    x_adv2 = x_natural.detach() - delta
    
    # Get teacher and model logits
    b_logits_T = teacher(x_natural)
    
    
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        x_adv2.requires_grad_()
        
        with torch.enable_grad():
            # Compute KL divergence losses
            edge2 = criterion_kl(F.log_softmax(model(x_adv2)/T, dim=1), F.softmax(b_logits_T/T, dim=1))
            edge3 = criterion_kl(F.log_softmax(model(x_adv)/T, dim=1), F.softmax(model(x_adv2)/T, dim=1))
            
            loss_kl = (1 - alpha) * edge2 + alpha * edge3
            
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        
        # Update adversarial examples with gradient sign
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    model.train()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)  # Clamp final adversarial examples
    return x_adv






'''Baseline'''

def APGD(model, x_natural, teacher, loss, T=30.0, alpha =0.9):
    '''
    EPS = 8/255
    ALPHA = 2/255
    STEPS = 10
    '''
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    step_size= 2/255  # 0.003
    epsilon= 8/255    # 0.031
    perturb_steps=10  # 10

    model.eval()
    # generate adversarial example
    delta = torch.empty_like(x_natural).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = x_natural.detach() + delta
    b_logits_T = teacher(x_natural)
    b_logits_S = model(x_natural)
    #x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                edge2 = criterion_kl(F.log_softmax(model(x_adv)/T, dim=1), F.softmax(b_logits_T/T, dim=1)) 
                edge3 = criterion_kl(F.log_softmax(model(x_adv)/T, dim=1), F.softmax(b_logits_S/T, dim=1)) 
                loss_kl = (1-alpha) * edge2 + alpha * edge3
    
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    model.train()

    return x_adv
