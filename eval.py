import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torchattacks.attacks.pgd import PGD
from TransferAttack.transferattack.gradient.pgn import PGN
from torchattacks.attacks.square import Square
from tqdm import tqdm
from data_augmentation import get_compound_aug


def eval_rob_acc(dataloader_test, model, device, dg, severity, method, dataset_name, eps):
    model.eval()
    steps = 20

    if dataset_name == "imagenet":
        epsilon = eps / 255
        alpha = 1 / 255
    elif dataset_name == "cifar10":
        epsilon = eps / 255
        alpha = 2 / 255

    if method == "PGD":
        attack = PGD(model, eps=epsilon, alpha=alpha, steps=steps)
    elif method == "PGN":
        attack = PGN(model, epsilon=8/255, alpha=2/255, beta=3.0, gamma=0.5,
                     num_neighbor=20, epoch=10, decay=1.)
    elif method == "Square":
        attack = Square(model, eps=8/255, n_queries=5000, n_restarts=1, p_init=.8)
    else:
        raise ValueError("Unknown method")

    corruption = get_compound_aug(severity) if dg == 1 else None
    total = 0
    correct1 = 0
    correct5 = 0

    for batch_idx, (ori_img, label) in enumerate(tqdm(dataloader_test, desc="Adv eval")):
        ori_img = ori_img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        if corruption is not None:
            ori_img = torch.stack([corruption(img) for img in ori_img]).to(device)

        # NO autocast here — adversarial generation must run in fp32
        adv_images = attack(ori_img, label)
        adv_images = adv_images.detach()

        # AMP is fine for the evaluation forward pass only
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits = model(adv_images)

        total += label.size(0)
        pred1 = logits.argmax(dim=1)
        correct1 += pred1.eq(label).sum().item()
        k = min(5, logits.size(1))
        _, pred5 = logits.topk(k, dim=1, largest=True, sorted=True)
        correct5 += pred5.eq(label.view(-1, 1)).any(dim=1).sum().item()

    rob_acc_top1 = 100.0 * correct1 / total
    rob_acc_top5 = 100.0 * correct5 / total
    return round(rob_acc_top1, 2), round(rob_acc_top5, 2)


def eval(model, device, loader, type, dg, severity):
    model.eval()
    loss = 0
    correct1 = 0   # top-1
    correct5 = 0   # top-5

    if dg == 1:
        corruption = get_compound_aug(severity)

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            if dg == 1:
                data = torch.stack([corruption(img) for img in data]).to(device)

            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()

            # ---- Top-1 accuracy ----
            pred1 = output.argmax(dim=1)
            correct1 += pred1.eq(target).sum().item()

            # ---- Top-5 accuracy ----
            _, pred5 = output.topk(5, dim=1, largest=True, sorted=True)
            correct5 += pred5.eq(target.view(-1, 1)).any(dim=1).sum().item()

    loss /= len(loader.dataset)
    top1_acc = 100. * correct1 / len(loader.dataset)
    top5_acc = 100. * correct5 / len(loader.dataset)

    print(
        '{}: Loss {:.4f}, Top-1 Acc: {:.2f}%, Top-5 Acc: {:.2f}%'.format(
            type, loss, top1_acc, top5_acc
        )
    )

    return loss, top1_acc, top5_acc



