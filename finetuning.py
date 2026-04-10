import os
import json
import argparse
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import models
from torchvision import models as vision_mod
from models.cifar10_models.normalize import Normalize
from models.imagenet_models.normalize import Normalize
from torch.utils.data import  DataLoader,Subset,Dataset
import numpy as np
from loss import trades_loss,tgra_loss,tradesU_loss,pgd_loss,tgra_loss_fgsm,mart_loss,dkl_finetune_loss,configure_dkl_finetune,reset_dkl_finetune_state
from util import load_model,set_seed,split_train_test_fixed_test,split_cifar10_fixed,split_dataset
from eval import eval,eval_rob_acc
import copy
from filelock import FileLock
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from models.cifar100_models.wideresnet import WideResNet
import torchvision.datasets as datasets
from sklearn.model_selection import StratifiedShuffleSplit
from data_augmentation import get_compound_aug
import math



def adjust_learning_rate_warmup(optimizer, epoch, lr_initial, max_epoch, warmup=2, min_lr=0.0):
    if epoch < warmup:
        lr = lr_initial * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / (max_epoch - warmup)
        lr = min_lr + 0.5 * (lr_initial - min_lr) * (1 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate(optimizer, epoch,lr_initial,dataset):
    if dataset=="imagenet":
        lr = lr_initial
        if epoch >= 10:
            lr = lr_initial * 0.1
        if epoch >= 20:
            lr = lr_initial * 0.01
    elif dataset=="cifar10":
        lr = lr_initial
        if epoch >= 10:               #10
            lr = lr_initial * 0.1
        if epoch >= 25:               #25
            lr = lr_initial * 0.01
        if epoch >= 30:               #30
            lr = lr_initial * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='VGG16',
                        choices=['VGG11', 'VGG16', 'VGG19', 'DenseNet161', 'InceptionV3', 'Resnet18', 'Resnet50','WideResnet','vit_b_16'])
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--loss', type=str, choices=['tgra','tradesU','tgra_fgsm','mart','dkl', 'trades','pgd'], default='tgra', help="Type of loss function to use.")
    parser.add_argument('--beta', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate')
    parser.add_argument('--scheduler', type=str, choices=['vanilla','cosineannealing', 'decay','decay_warmup'], default='vanilla')
    parser.add_argument('--eps', type=int, default=8)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--num_step', type=int, default=1)
    parser.add_argument('--frq_test', type=int, default=0)
    parser.add_argument('--severity', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--corruption', type=int, default=0)
    parser.add_argument('--split', type=int, default=50)
    return parser.parse_args()

def train(type_loss, N, model, reference_model, device, train_loader, optimizer, epoch,beta,eps,step_size,dg,severity,args):
    model.train()
    epsilon=eps/255
    num_steps=N
    step_size=step_size/255
    loss=None
    loss_value=0
    if dg==1:
        corruption = get_compound_aug(severity)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if dg==1:
            data=torch.stack([corruption(img) for img in data]).to(device)

        optimizer.zero_grad()
        
        if type_loss=="trades":
            # calculate robust loss
            loss = trades_loss(model=model,
                            x_natural=data,
                            y=target,
                            optimizer=optimizer,
                            step_size=step_size,
                            epsilon=epsilon,
                            perturb_steps=num_steps,
                            beta=beta)
        elif type_loss=="tgra":
            loss = tgra_loss(model=model,
                           reference_model=reference_model,
                            x_natural=data,
                            y=target,
                            optimizer=optimizer,
                            step_size=step_size,
                            epsilon=epsilon,
                            perturb_steps=num_steps,
                            beta=beta)
        elif type_loss=="tgra_fgsm":
            loss = tgra_loss_fgsm(model=model,
                           reference_model=reference_model,
                            x_natural=data,
                            y=target,
                            optimizer=optimizer,
                            epsilon=epsilon,
                            step_size=step_size,
                            beta=beta)
        elif type_loss=="tradesU":
            loss = tradesU_loss(model=model,
                           reference_model=reference_model,
                            x_natural=data,
                            y=target,
                            optimizer=optimizer,
                            step_size=step_size,
                            epsilon=epsilon,
                            perturb_steps=num_steps,
                            beta=beta)
        elif type_loss=="mart":
            loss=mart_loss(model=model,
                            x_natural=data,
                            y=target,
                            optimizer=optimizer,
                            step_size=step_size,
                            epsilon=epsilon,
                            perturb_steps=num_steps,
                            beta=beta)
        elif type_loss=='dkl':
            ncls = 1000 if args.dataset == "imagenet" else 10
            loss = dkl_finetune_loss(
                model=model,
                x_natural=data,
                y=target,
                optimizer=optimizer,
                step_size=step_size,
                epsilon=epsilon,
                perturb_steps=num_steps,
                beta=beta,
                epoch=epoch,
                total_epochs=args.epoch,
                num_classes=ncls,
            )
        else:
            loss = pgd_loss(model=model,
                            x_natural=data,
                            y=target,
                            optimizer=optimizer,
                            step_size=step_size,
                            epsilon=epsilon,
                            perturb_steps=num_steps)
        loss.backward()
        optimizer.step()
        loss_value+=loss.item()
        # print progress
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    return loss_value

def train_finetuning(args,device):
    model_name,type_loss,N,beta,lr,scheduler_flag=args.model,args.loss,args.num_step,args.beta,args.lr,args.scheduler
    model=None
    if args.dataset=="cifar10":
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])


        transform_test = transforms.Compose([transforms.ToTensor()])

        if args.split==0:
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
            testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)
            print(f"Train size: {len(trainset)} ")
            print(f"Fixed test: {len(testset)} ")
        else:

            split_base = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=None)
            split_train=args.split/100

            train_idx, fixed_test_idx = split_cifar10_fixed(split_base,train_frac_of_full=split_train,test_frac=0.50,seed_test=42,seed_train=123)

            trainset = Subset(torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_train),train_idx)
            testset = Subset(torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_test),fixed_test_idx)

            trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
            testloader  = DataLoader(testset,  batch_size=256, shuffle=False, num_workers=4)

            print(f"Train size: {len(trainset)} ({split_train:.0%} of full CIFAR10 test)")
            print(f"Fixed test: {len(testset)} ")

        model = load_model(model_name, True)
        model = nn.Sequential(models.cifar10_models.normalize.Normalize(), model)
        if "WideResnet" in model_name:
            #model.load_state_dict(torch.load('./models/cifar10_models/state_dicts/WideResnet_ft_kl_6_10_0.01_decay.pt'))
            model.load_state_dict(torch.load(f'./models/cifar10_models/state_dicts/{model_name}.pt'))

    elif args.dataset=="imagenet":

        transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ])

        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        #elif model_name=="vit_b_16":
        #    transform_train = transforms.Compose([
        #        transforms.RandomResizedCrop(384, interpolation=InterpolationMode.BICUBIC),
        #        transforms.RandomHorizontalFlip(p=0.5),
        #        transforms.RandomRotation(15),  
        #        transforms.ToTensor(),
        #    ])
            
        #    transform_test = transforms.Compose([
        #        transforms.Resize(384, interpolation=InterpolationMode.BICUBIC),
        #        transforms.CenterCrop(384),
        #        transforms.ToTensor(),
        #    ])

        imagenet_path = "/home/datasets/Imagenet"

        #split_base = datasets.ImageNet(root=imagenet_path, split="val", transform=None)
        #split_train=args.split/100
        #train_idx_subset, test_idx_subset = split_train_test_fixed_test(split_base,train_frac=split_train,test_frac=0.10)

        # Create datasets with independent transforms
        #train_ds = datasets.ImageNet(root=imagenet_path, split="val", transform=transform_train)
        #test_ds = datasets.ImageNet(root=imagenet_path, split="val", transform=transform_test)

        #trainset = torch.utils.data.Subset(train_ds, train_idx_subset.indices)
        #testset  = torch.utils.data.Subset(test_ds,  test_idx_subset.indices)

        #trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
        #testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

        #print(f"Train size: {len(trainset)} ({split_train:.0%})")
        #print(f"Test size : {len(testset)} (10%)")
        split_train=args.split/100
        split_test=1-split_train
        val_dataset = datasets.ImageNet(root=imagenet_path, split='val', transform=None)
        trainset,testset=split_dataset(val_dataset,[split_train,split_test])
        trainset.dataset.transform = transform_train
        testset.dataset.transform = transform_test
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

        if model_name=="Resnet50":
            model = vision_mod.resnet50(weights=vision_mod.ResNet50_Weights.IMAGENET1K_V2)
        elif model_name=="vit_b_16":
            model = vision_mod.vit_b_16(weights=vision_mod.ViT_B_16_Weights.IMAGENET1K_V1)
        model = nn.Sequential(models.imagenet_models.normalize.Normalize(), model)
    
    model=model.to(device)
    copy_not=['trades','pgd','mart','dkl']

    if type_loss not in copy_not:
        reference_model = copy.deepcopy(model)
    else:
        reference_model=None
    
    
    for name, param in model[1].named_parameters():
        param.requires_grad = True

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=2e-4)
    scheduler=None
    if scheduler_flag=="cosineannealing":
        print("Activate scheduler")
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-4)

    
    if type_loss == 'dkl':
        reset_dkl_finetune_state()
        configure_dkl_finetune(total_epochs=args.epoch, train_budget='high', alpha=4.0, beta_internal=20.0, gamma=1.0, prior_temperature=4.0, distance='l_inf')


    epochs=args.epoch
    list_test_acc_top1=[]
    list_test_acc_top5=[]
    list_test_rob_acc_top1=[]
    list_test_rob_acc_top5=[]
    list_train_acc=[]
    list_train_rob_acc=[]
    print("PRETRAINED MODEL RESULTS: ")
    _, train_accuracy,_=eval(model, device, trainloader,'Training',args.corruption,args.severity)
    train_rob_acc,_=eval_rob_acc(trainloader,model,device,args.corruption,args.severity,"PGD",args.dataset,args.eps)
    _, test_accuracy_top1,test_accuracy_top5=eval(model, device, testloader,'Test',args.corruption,args.severity)
    rob_acc_top1,rob_acc_top5=eval_rob_acc(testloader,model,device,args.corruption,args.severity,"PGD",args.dataset,args.eps)
    print("Natural Test Accuracy top1: ",test_accuracy_top1)
    print("Robust Test Accuracy top1: ",rob_acc_top1)
    list_test_acc_top1.append(test_accuracy_top1)
    list_test_acc_top5.append(test_accuracy_top5)
    list_test_rob_acc_top1.append(rob_acc_top1)
    list_test_rob_acc_top5.append(rob_acc_top5)
    list_train_acc.append(train_accuracy)
    list_train_rob_acc.append(train_rob_acc)
    os.makedirs(f"./models/{args.dataset}_models/state_dicts", exist_ok=True)
    for epoch in range(epochs):

        if scheduler_flag=='decay':
            adjust_learning_rate(optimizer, epoch,lr,args.dataset)
        elif scheduler_flag=='decay_warmup':
            adjust_learning_rate_warmup(optimizer, epoch, lr, epochs)

        _=train(type_loss,N, model, reference_model, device, trainloader, optimizer, epoch,beta,args.eps,args.step_size,args.corruption,args.severity,args)
        if epoch % args.frq_test==0 or epoch==epochs-1:
            print('================================================================')
            _, train_accuracy,_=eval(model, device, trainloader,'Training',args.corruption,args.severity)
            _, test_accuracy_top1,test_accuracy_top5=eval(model, device, testloader,'Test',args.corruption,args.severity)
            print("Natural Test Accuracy top1: ",test_accuracy_top1)
            train_rob_acc,_=eval_rob_acc(trainloader,model,device,args.corruption,args.severity,"PGD",args.dataset,args.eps)
            rob_acc_top1,rob_acc_top5=eval_rob_acc(testloader,model,device,args.corruption,args.severity,"PGD",args.dataset,args.eps)
            print("Robust Test Accuracy top1: ",rob_acc_top1)
            print('================================================================')
            list_train_acc.append(train_accuracy)
            list_train_rob_acc.append(train_rob_acc)
            list_test_acc_top1.append(test_accuracy_top1)
            list_test_acc_top5.append(test_accuracy_top5)
            list_test_rob_acc_top1.append(rob_acc_top1)
            list_test_rob_acc_top5.append(rob_acc_top5)
        if scheduler_flag=="cosineannealing":
            scheduler.step()
    torch.save(model.state_dict(), f"./models/{args.dataset}_models/state_dicts/{args.model}_corruption_{args.corruption}_sev_{args.severity}_{args.loss}_beta_{args.beta}_eps_{args.eps}_alpha_{args.step_size}_step_{args.num_step}_{args.lr}_{args.scheduler}_split_{args.split}.pt")
    return list_test_acc_top1,list_test_acc_top5,list_test_rob_acc_top1,list_test_rob_acc_top5,list_train_acc,list_train_rob_acc

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print("START finetuning param: ")

    list_test_acc_top1,list_test_acc_top5,list_test_rob_acc_top1,list_test_rob_acc_top5,list_train_acc,list_train_rob_acc=train_finetuning(args,device)

    filename = f"result_finetuning_{args.dataset}_test.json"
    lock = FileLock(filename + ".lock")

    with lock:
        if os.path.exists(filename):
            with open(filename, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        entry = {
                "name_test": f"{args.model}_corruption_{args.corruption}_sev_{args.severity}_"
                            f"{args.loss}_beta_{args.beta}_eps_{args.eps}_alpha_{args.step_size}_step_{args.num_step}_{args.lr}_{args.scheduler}_split_{args.split}",
                "nat_test_acc_top1": list_test_acc_top1,
                "nat_test_acc_top5": list_test_acc_top5,
                "rob_test_acc_top1": list_test_rob_acc_top1,
                "rob_test_acc_top5": list_test_rob_acc_top5,
                "nat_train_acc":list_train_acc,
                "rob_train_acc":list_train_rob_acc
            }

        data.append(entry)

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)