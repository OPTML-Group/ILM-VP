import os
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

import sys
sys.path.append(".")
from datatools.const import IMAGENETNORMALIZE
from datatools.prepare_data import prepare_clip_data
from tools.misc import gen_folder_name, set_seed
from cfg import *


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--network', choices=["resnet18", "resnet50", "instagram"], required=True)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--dataset', choices=["cifar10", "cifar100", "abide", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], default='cifar10')
    p.add_argument('--epoch', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.01)
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    exp = f"finetune-resnet"
    save_path = os.path.join(results_path, exp, gen_folder_name(args))
    preprocess = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
    ])
    loaders, class_names = prepare_clip_data(args.dataset, data_path=data_path, preprocess=preprocess)
    
    # Network
    if args.network == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights
        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    elif args.network == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    elif args.network == "instagram":
        from torch import hub
        network = hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl').to(device)
    else:
        raise NotImplementedError(f"{args.network} is not supported")
    network.requires_grad_(True)
    network = network.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=5e-4)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5*args.epoch), int(0.72*args.epoch)], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    # Make Dir
    os.makedirs(save_path, exist_ok=True)
    logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    # Train
    best_acc = 0. 
    scaler = GradScaler()
    for epoch in range(args.epoch):
        network.train()
        total_num = 0
        true_num = 0
        loss_sum = 0
        pbar = tqdm(loaders['train'], total=len(loaders['train']),
                desc=f"Epo {epoch} Training Lr {optimizer.param_groups[0]['lr']:.1e}", ncols=100)
        for x, y in pbar:
            if x.get_device() == -1:
                x, y = x.to(device), y.to(device)
            pbar.set_description_str(f"Epo {epoch} Training Lr {optimizer.param_groups[0]['lr']:.1e}", refresh=True)
            optimizer.zero_grad()
            with autocast():
                fx = network(x)
                loss = F.cross_entropy(fx, y, reduction='mean')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            loss_sum += loss.item() * fx.size(0)
            pbar.set_postfix_str(f"Acc {100*true_num/total_num:.2f}%")
        scheduler.step()
        logger.add_scalar("train/acc", true_num/total_num, epoch)
        logger.add_scalar("train/loss", loss_sum/total_num, epoch)
        
        # Test
        network.eval()
        total_num = 0
        true_num = 0
        pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Epo {epoch} Testing", ncols=100)
        for x, y in pbar:
            if x.get_device() == -1:
                x, y = x.to(device), y.to(device)
            with torch.no_grad():
                fx = network(x)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            acc = true_num/total_num
            pbar.set_postfix_str(f"Acc {100*acc:.2f}%")
        logger.add_scalar("test/acc", acc, epoch)

        # Save CKPT
        state_dict = {
            "fc_dict": network.fc.state_dict(),
            "optimizer_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        }
        if acc > best_acc:
            best_acc = acc
            state_dict['best_acc'] = best_acc
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))
        torch.save(state_dict, os.path.join(save_path, 'ckpt.pth'))
    