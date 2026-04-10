import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def mart_loss(
    model,
    x_natural,
    y,
    optimizer,
    step_size: float = 0.007,
    epsilon: float = 0.031,
    perturb_steps: int = 10,
    beta: float = 6.0,
    distance: str = 'l_inf'
) -> torch.Tensor:
    
    # Constants
    EPS_STABLE = 1e-12
    CLAMP_STABLE = 1.0001
    WEIGHT_STABLE = 1.0000001

    kl = nn.KLDivLoss(reduction='none')
    batch_size = len(x_natural)

    # ── Adversarial example generation ──────────────────────────────────────
    model.eval()
    with torch.no_grad():
        x_adv = x_natural + 0.001 * torch.randn_like(x_natural)

    if distance == 'l_inf':
        x_adv = x_adv.detach().requires_grad_(True)
        for _ in range(perturb_steps):
            loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, x_adv)[0]

            with torch.no_grad():
                x_adv = x_adv + step_size * grad.sign()
                x_adv = torch.clamp(x_adv, x_natural - epsilon, x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

            if _ < perturb_steps - 1:          # skip on last iteration
                x_adv = x_adv.requires_grad_(True)
    else:
        with torch.no_grad():
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    # ── Loss computation ─────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    # Single forward pass, split after
    x_both  = torch.cat([x_natural, x_adv.detach()], dim=0)
    logits_both = model(x_both)
    logits, logits_adv = logits_both.chunk(2, dim=0)

    # Adversarial loss
    adv_probs   = F.softmax(logits_adv, dim=1)
    top2_idx    = torch.topk(adv_probs, k=2, dim=1).indices   # faster than argsort
    new_y       = torch.where(top2_idx[:, 0] == y, top2_idx[:, 1], top2_idx[:, 0])

    loss_adv = (
        F.cross_entropy(logits_adv, y)
        + F.nll_loss(torch.log(CLAMP_STABLE - adv_probs + EPS_STABLE), new_y)
    )

    # Robust loss
    nat_probs  = F.softmax(logits, dim=1)
    true_probs = nat_probs[torch.arange(batch_size), y]        # avoids gather+squeeze

    kl_per_sample = kl(torch.log(adv_probs + EPS_STABLE), nat_probs).sum(dim=1)
    loss_robust   = (kl_per_sample * (WEIGHT_STABLE - true_probs)).mean()  # mean == /batch_size

    return loss_adv + beta * loss_robust

def pgd_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10):
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    x_adv = x_natural + torch.empty_like(x_natural).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)

    for _ in range(perturb_steps):
            x_adv.requires_grad_()
            logits = model(x_adv)
            with torch.enable_grad():
                loss = loss_fn(logits, y)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()
    outputs = model(x_adv)
    loss = loss_fn(outputs, y)
    return loss

def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss



def tgra_loss(model,
            reference_model,
            x_natural,
            y,
            optimizer,
            step_size=0.003,
            epsilon=0.031,
            perturb_steps=10,
            beta=1.0):
    
    criterion_kl = nn.KLDivLoss(size_average=False)

    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    with torch.no_grad():
        ref_prob = F.softmax(reference_model(x_natural), dim=1)


    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).cuda()
    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        # No need for torch.enable_grad() inside model.eval() loop
        # if outer context has no_grad, but being explicit is fine
        loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), ref_prob)
        grad = torch.autograd.grad(loss_kl, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + step_size * grad.sign()
            x_adv = torch.clamp(x_adv, x_natural - epsilon, x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    optimizer.zero_grad()

    loss_natural_kl = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_natural), dim=1),
                                                    ref_prob)  
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                  ref_prob)
    loss = loss_natural_kl + beta * loss_robust
    return loss

def tgra_loss_fgsm(model,
            reference_model,
            x_natural,
            y,
            optimizer,
            epsilon=0.031,
            step_size=1,
            beta=1.0):
    
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    with torch.no_grad():
        ref_prob = F.softmax(reference_model(x_natural), dim=1)
    x_adv = x_natural + torch.empty_like(x_natural).uniform_(-epsilon, epsilon)
    #x_adv = torch.clamp(x_adv, min=0, max=1).detach()
    x_adv.requires_grad_()
    with torch.enable_grad():
            #previous
            #loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
            #                           F.softmax(model(x_natural), dim=1))
        loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       ref_prob)
    grad = torch.autograd.grad(loss_kl, [x_adv])[0]
    x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
    x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    #logits = model(x_natural)
    loss_natural_kl = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_natural), dim=1),
                                                    ref_prob)
    
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                   ref_prob)
    loss = loss_natural_kl + beta * loss_robust
    return loss


def tradesU_loss(model,
            reference_model,
            x_natural,
            y,
            optimizer,
            step_size=0.003,
            epsilon=0.031,
            perturb_steps=10,
            beta=1.0):
    
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)

    with torch.no_grad():
        ref_prob = F.softmax(reference_model(x_natural), dim=1)
        # Also cache clean student logits for PGD inner loop
        nat_logits_cached = model(x_natural)
        nat_prob_cached   = F.softmax(nat_logits_cached, dim=1)



    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).cuda()
    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        # Use cached clean probs — they don't change during PGD
        loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), nat_prob_cached)
        grad = torch.autograd.grad(loss_kl, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + step_size * grad.sign()
            x_adv = torch.clamp(x_adv, x_natural - epsilon, x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    # zero gradient
    optimizer.zero_grad()
    logit_clean_student=model(x_natural)
    # calculate robust loss
    #logits = model(x_natural)
    loss_natural_kl = (1.0 / batch_size) * criterion_kl(F.log_softmax(logit_clean_student, dim=1),
                                                    F.softmax(reference_model(x_natural), dim=1))
    
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(logit_clean_student.detach(), dim=1))
    loss = loss_natural_kl + beta * loss_robust
    return loss





# =========================
# DKL finetuning utilities
# =========================
_DKL_FT_STATE = {
    "weight": None,
    "accum": None,
    "num_classes": None,
    "device": None,
    "last_epoch": None,
    "total_epochs": 30,
    "train_budget": "high",
    "alpha": 4.0,
    "beta_internal": 20.0,
    "gamma": 1.0,
    "prior_temperature": 4.0,
    "distance": "l_inf",
}


def reset_dkl_finetune_state():
    _DKL_FT_STATE["weight"] = None
    _DKL_FT_STATE["accum"] = None
    _DKL_FT_STATE["num_classes"] = None
    _DKL_FT_STATE["device"] = None
    _DKL_FT_STATE["last_epoch"] = None



def configure_dkl_finetune(
    total_epochs=30,
    train_budget='high',
    alpha=4.0,
    beta_internal=20.0,
    gamma=1.0,
    prior_temperature=4.0,
    distance='l_inf',
):
    _DKL_FT_STATE["total_epochs"] = max(int(total_epochs), 1)
    _DKL_FT_STATE["train_budget"] = train_budget
    _DKL_FT_STATE["alpha"] = float(alpha)
    _DKL_FT_STATE["beta_internal"] = float(beta_internal)
    _DKL_FT_STATE["gamma"] = float(gamma)
    _DKL_FT_STATE["prior_temperature"] = float(prior_temperature)
    _DKL_FT_STATE["distance"] = distance



def _maybe_init_dkl_state(num_classes, device, epoch):
    needs_init = (
        _DKL_FT_STATE["weight"] is None
        or _DKL_FT_STATE["num_classes"] != num_classes
        or _DKL_FT_STATE["device"] != str(device)
    )
    if needs_init:
        weight = torch.ones(num_classes, num_classes, device=device) / float(num_classes)
        accum = torch.zeros(num_classes, num_classes, device=device)
        _DKL_FT_STATE["weight"] = weight
        _DKL_FT_STATE["accum"] = accum
        _DKL_FT_STATE["num_classes"] = num_classes
        _DKL_FT_STATE["device"] = str(device)
        _DKL_FT_STATE["last_epoch"] = int(epoch)
        return

    if epoch != _DKL_FT_STATE["last_epoch"]:
        prev_weight = _DKL_FT_STATE["weight"]
        accum = _DKL_FT_STATE["accum"]
        row_sums = accum.sum(dim=1, keepdim=True)
        new_weight = prev_weight.clone()
        valid_rows = row_sums.squeeze(1) > 0
        if valid_rows.any():
            new_weight[valid_rows] = accum[valid_rows] / row_sums[valid_rows]
        _DKL_FT_STATE["weight"] = new_weight
        _DKL_FT_STATE["accum"] = torch.zeros_like(accum)
        _DKL_FT_STATE["last_epoch"] = int(epoch)



#def _dkl_attack_schedule(epoch, epsilon, max_steps):
#    total_epochs = max(int(_DKL_FT_STATE["total_epochs"]), 1)
#    progress = float(epoch + 1) / float(total_epochs)
#    varepsilon = epsilon * progress
#    max_steps = max(int(max_steps), 2)

#    if _DKL_FT_STATE["train_budget"] == 'low':
#        return varepsilon, varepsilon, 2

 #   epoch_scale = total_epochs / 200.0
 #   steps = 2
 #   step_size = varepsilon
 #   if epoch + 1 <= int(round(50 * epoch_scale)):
 #       steps = 2
 #       step_size = varepsilon
 #   elif epoch + 1 <= int(round(100 * epoch_scale)):
 #       steps = min(3, max_steps)
 #       step_size = 2.0 * varepsilon / 3.0
 #   elif epoch + 1 <= int(round(150 * epoch_scale)):
 #       steps = min(4, max_steps)
 #       step_size = varepsilon / 2.0
 #   else:
 #       steps = min(5, max_steps)
 #       step_size = varepsilon / 2.0
 #   return varepsilon, step_size, steps

def dkl_attack_schedule(epoch, total_epochs, epsilon, train_budget='high'):
    """
    DKL attack schedule for finetuning/training.

    Args:
        epoch:        current epoch index, assumed 0-based
        total_epochs: total number of training epochs
        epsilon:      final perturbation budget in normalized units
                      (e.g. 8/255, not 8)
        train_budget: 'low' or 'high'

    Returns:
        varepsilon:   epoch-ramped epsilon
        step_size:    PGD step size for this epoch
        iters_attack: number of PGD steps for this epoch
    """
    e = epoch + 1  # convert to 1-based for smoother ramp

    # ramp epsilon across training, following original DKL idea
    varepsilon = epsilon * e / total_epochs

    if train_budget == 'low':
        step_size = varepsilon
        iters_attack = 2
        return varepsilon, step_size, iters_attack

    # high-budget progressive schedule compressed to total_epochs
    q1 = int(0.25 * total_epochs)
    q2 = int(0.50 * total_epochs)
    q3 = int(0.75 * total_epochs)

    # keep thresholds valid for very small total_epochs
    q1 = max(q1, 1)
    q2 = max(q2, q1 + 1)
    q3 = max(q3, q2 + 1)

    if e <= q1:
        step_size = varepsilon
        iters_attack = 2
    elif e <= q2:
        step_size = 2.0 * varepsilon / 3.0
        iters_attack = 3
    elif e <= q3:
        step_size = varepsilon / 2.0
        iters_attack = 4
    else:
        step_size = varepsilon / 2.0
        iters_attack = 5

    return varepsilon, step_size, iters_attack



def dkl_loss_original(logits_student, logits_teacher, temperature=1.0, alpha=1.0, beta=1.0, gamma=1.0, CLASS_PRIOR=None):
    num_classes = logits_teacher.size(1)
    delta_n = logits_teacher.view(-1, num_classes, 1) - logits_teacher.view(-1, 1, num_classes)
    delta_a = logits_student.view(-1, num_classes, 1) - logits_student.view(-1, 1, num_classes)

    assert CLASS_PRIOR is not None, 'CLASS PRIOR information should be collected for DKL'
    with torch.no_grad():
        class_prior = torch.pow(CLASS_PRIOR, gamma)
        p_n = class_prior.view(-1, num_classes, 1) @ class_prior.view(-1, 1, num_classes)

    loss_mse = 0.25 * (torch.pow(delta_n - delta_a, 2) * p_n).sum() / p_n.sum().clamp_min(1e-12)
    loss_sce = -(F.softmax(logits_teacher / temperature, dim=1).detach() * F.log_softmax(logits_student / temperature, dim=-1)).sum(1).mean()
    return beta * loss_mse + alpha * loss_sce



def perturb_input_dkl(
    model,
    x_natural,
    step_size=0.003,
    epsilon=0.031,
    perturb_steps=10,
    distance='l_inf',
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
    CLASS_PRIOR=None,
):
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = dkl_loss_original(
                    model(x_adv),
                    model(x_natural),
                    CLASS_PRIOR=CLASS_PRIOR,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                )
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()



def dkl_finetune_loss(
    model,
    x_natural,
    y,
    optimizer,
    step_size=None,
    epsilon=0.031,
    perturb_steps=None,
    beta=20.0,
    epoch=0,
    total_epochs=30,
    train_budget='high',
    alpha=4.0,
    gamma=1.0,
    temperature=4.0,
    num_classes=10,
):
    model.eval()
    device = x_natural.device

    if not hasattr(dkl_finetune_loss, "weight"):
        dkl_finetune_loss.weight = torch.ones(num_classes, num_classes, device=device) / num_classes

    weight = dkl_finetune_loss.weight.to(device)
    WEIGHT = torch.zeros(num_classes, num_classes, device=device)

    with torch.no_grad():
        onehot = F.one_hot(y, num_classes).float()
        sample_weight = onehot @ weight

    attack_epsilon, attack_step, attack_iters = dkl_attack_schedule(
        epoch=epoch,
        total_epochs=total_epochs,
        epsilon=epsilon,
        train_budget=train_budget,
    )

    if step_size is not None:
        attack_step = step_size
    if perturb_steps is not None:
        attack_iters = perturb_steps

    x_adv = perturb_input_dkl(
        model=model,
        x_natural=x_natural,
        step_size=attack_step,
        epsilon=attack_epsilon,
        perturb_steps=attack_iters,
        alpha=1.0,
        beta=0.0,
        gamma=1.0,
        CLASS_PRIOR=sample_weight,
    )

    model.train()
    optimizer.zero_grad()

    logits_adv = model(x_adv)
    logits_nat = model(x_natural)

    with torch.no_grad():
        WEIGHT += onehot.t() @ F.softmax(logits_nat.detach() / temperature, dim=-1)
        row_sum = WEIGHT.sum(dim=1, keepdim=True).clamp_min(1e-12)
        dkl_finetune_loss.weight = WEIGHT / row_sum

    loss_robust = dkl_loss_original(
        logits_adv,
        logits_nat,
        CLASS_PRIOR=sample_weight,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        temperature=temperature,
    )
    loss_natural = F.cross_entropy(logits_nat, y)

    return loss_natural + loss_robust