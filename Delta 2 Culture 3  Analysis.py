
'''Delta 2 Culture 3  Analysis'''


''' 50.4 '''
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
    adv1 = x_natural.detach() + delta
    adv2 = x_natural.detach() - delta
    
    # Get teacher and model logits
    b_logits_T = teacher(x_natural)
    
    for _ in range(perturb_steps):
        adv1.requires_grad_()
        adv2.requires_grad_()
        
        with torch.enable_grad():
            # Compute KL divergence losses
            edge2 = criterion_kl(F.log_softmax(model(adv2)/T, dim=1), F.softmax(b_logits_T/T, dim=1))
            edge3 = criterion_kl(F.log_softmax(model(adv1)/T, dim=1), F.softmax(model(adv2)/T, dim=1))
            
            loss_kl = (1 - alpha) * edge2 + alpha * edge3
            
        grad = torch.autograd.grad(loss_kl, [adv1])[0]
        
        # Update adversarial examples with gradient sign
        adv1 = adv1.detach() + step_size * torch.sign(grad.detach())
        adv1 = torch.min(torch.max(adv1, x_natural - epsilon), x_natural + epsilon)
        adv1 = torch.clamp(adv1, 0.0, 1.0)
    
    model.train()
    #adv1 = torch.clamp(adv1, 0.0, 1.0)  # Clamp final adversarial examples
    # Compute the average of adv1 and adv2 to get xs
    xs = (adv1 + adv2) / 2.0
    
    # Compute delta as (adv1 - adv2) / 2.0
    delta = (adv1 - adv2) / 2.0
    
    adv = xs + delta
    adv = torch.clamp(adv, 0.0, 1.0)
    return adv




'''  49.96  '''
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
    adv1 = x_natural.detach() + delta
    adv2 = x_natural.detach() + delta
    
    # Get teacher and model logits
    b_logits_T = teacher(x_natural)
    
    for _ in range(perturb_steps):
        adv1.requires_grad_()
        adv2.requires_grad_()
        
        with torch.enable_grad():
            # Compute KL divergence losses
            edge2 = criterion_kl(F.log_softmax(model(adv2)/T, dim=1), F.softmax(b_logits_T/T, dim=1))
            edge3 = criterion_kl(F.log_softmax(model(adv1)/T, dim=1), F.softmax(model(adv2)/T, dim=1))
            
            loss_kl = (1 - alpha) * edge2 + alpha * edge3
            
        grad = torch.autograd.grad(loss_kl, [adv1, adv2])[0]
        
        # Update adversarial examples with gradient sign
        adv1 = adv1.detach() + step_size * torch.sign(grad.detach())
        adv1 = torch.min(torch.max(adv1, x_natural - epsilon), x_natural + epsilon)
        adv1 = torch.clamp(adv1, 0.0, 1.0)
    
    model.train()
    #adv1 = torch.clamp(adv1, 0.0, 1.0)  # Clamp final adversarial examples
    # Compute the average of adv1 and adv2 to get xs
    xs = (adv1 + adv2) / 2.0
    
    # Compute delta as (adv1 - adv2) / 2.0
    delta = (adv1 - adv2) / 2.0
    
    adv = xs + delta
    adv = torch.clamp(adv, 0.0, 1.0)
    return adv



''' 49.92 '''
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
    adv1 = x_natural.detach() + delta
    adv2 = x_natural.detach() - delta
    
    # Get teacher and model logits
    b_logits_T = teacher(x_natural)
    a_logits_S = model(adv2)
    for _ in range(perturb_steps):
        adv1.requires_grad_()
        #
        
        with torch.enable_grad():
            # Compute KL divergence losses
            edge2 = criterion_kl(F.log_softmax(a_logits_S/T, dim=1), F.softmax(b_logits_T/T, dim=1))
            edge3 = criterion_kl(F.log_softmax(model(adv1)/T, dim=1), F.softmax(a_logits_S/T, dim=1))
            
            loss_kl = (1 - alpha) * edge2 + alpha * edge3
            
        grad = torch.autograd.grad(loss_kl, [adv1])[0]
        
        # Update adversarial examples with gradient sign
        adv1 = adv1.detach() + step_size * torch.sign(grad.detach())
        adv1 = torch.min(torch.max(adv1, x_natural - epsilon), x_natural + epsilon)
        adv1 = torch.clamp(adv1, 0.0, 1.0)
    
    model.train()
    return adv1


'''49.84'''
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
    adv1 = x_natural.detach() + delta
    adv2 = x_natural.detach() - delta
    adv2 = torch.clamp(adv2, 0.0, 1.0)
    # Get teacher and model logits
    a_logits_S = model(adv2)
    for _ in range(perturb_steps):
        adv1.requires_grad_()
        
        with torch.enable_grad():
            # Compute KL divergence losses
            edge3 = criterion_kl(F.log_softmax(model(adv1)/T, dim=1), F.softmax(a_logits_S/T, dim=1))
            
            loss_kl = edge3
            
        grad = torch.autograd.grad(loss_kl, [adv1])[0]
        
        # Update adversarial examples with gradient sign
        adv1 = adv1.detach() + step_size * torch.sign(grad.detach())
        adv1 = torch.min(torch.max(adv1, x_natural - epsilon), x_natural + epsilon)
        adv1 = torch.clamp(adv1, 0.0, 1.0)
    
    model.train()
    return adv1




def APGD2(model, x_natural, T=30.0):
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
    adv1 = x_natural.detach() + delta
    adv2 = x_natural.detach() - delta
    # Get teacher and model logits
    for _ in range(perturb_steps):
        adv1.requires_grad_()
        
        with torch.enable_grad():
            # Compute KL divergence losses
            edge3 = criterion_kl(F.log_softmax(model(adv1)/T, dim=1), F.softmax(model(x_natural)/T, dim=1))
            
            loss_kl = edge3
            
        grad = torch.autograd.grad(loss_kl, [adv1])[0]
        
        # Update adversarial examples with gradient sign
        adv1 = adv1.detach() + step_size * torch.sign(grad.detach())
        adv1 = torch.min(torch.max(adv1, x_natural - epsilon), x_natural + epsilon)
        adv1 = torch.clamp(adv1, 0.0, 1.0)
        
    model.train()
    return adv1