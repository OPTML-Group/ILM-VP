from functools import partial
import os
import numpy as np
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import io
from PIL import Image

import sys
sys.path.append(".")
from datatools.prepare_data import prepare_ar_resnet_data
from datatools.const import IMAGENETCLASSES, IMAGENETNORMALIZE
from tools.frequency_mapping import get_dist_matrix
from tools.misc import gen_folder_name, set_seed
from tools.draw_mapping import plot_mapping
from models.adv_program import AdvProgram
from cfg import *


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--dataset', choices=["cifar10", "cifar100", "abide", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], default='cifar10')
    p.add_argument('--epoch', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.01)
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    mapping_method = "solo" # Use solo
    assert mapping_method in ["solo", "mean", "max"]
    if mapping_method == 'solo':
        mapping_num = 1
    def label_mapping_base(logits, mapping_sequence):
        if mapping_method == "mean":
            modified_logits = logits[:, mapping_sequence].reshape(logits.size(0), -1, mapping_num).mean(-1)
        elif mapping_method == "max":
            modified_logits = logits[:, mapping_sequence].reshape(logits.size(0), -1, mapping_num).max(-1).values
        else:
            modified_logits = logits[:, mapping_sequence]
        return modified_logits

    exp = f"baseline-label-mapping-resnet"
    save_path = os.path.join(results_path, exp, gen_folder_name(args))
    loaders, configs = prepare_ar_resnet_data(args.dataset, data_path=data_path)
    normalize = transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std'])

    # Network
    network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    network.requires_grad_(False)
    network = network.to(device)
    network.eval()

    # Adversarial Program
    adv_program = AdvProgram(224, mask=configs['mask'], normalize=normalize).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(adv_program.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5*args.epoch), int(0.72*args.epoch)], gamma=0.1)

    # Make Dir
    os.makedirs(save_path, exist_ok=True)
    logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    
    # Label Mapping
    mapping_sequence = torch.randperm(1000)[:len(configs['class_names'])]
    label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)

    # Train
    best_acc = 0. 
    scaler = GradScaler()
    for epoch in range(args.epoch):
        adv_program.train()
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
                fx = label_mapping(network(adv_program(x)))
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
        adv_program.eval()
        total_num = 0
        true_num = 0
        pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Epo {epoch} Testing", ncols=100)
        fx0s = []
        ys = []
        for x, y in pbar:
            if x.get_device() == -1:
                x, y = x.to(device), y.to(device)
            ys.append(y)
            with torch.no_grad():
                fx0 = network(adv_program(x))
                fx = label_mapping(fx0)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            acc = true_num/total_num
            fx0s.append(fx0)
            pbar.set_postfix_str(f"Acc {100*acc:.2f}%")
        fx0s = torch.cat(fx0s).cpu()
        ys = torch.cat(ys).cpu()
        mapping_matrix = get_dist_matrix(fx0s, ys)
        with io.BytesIO() as buf:
            plot_mapping(mapping_matrix, mapping_sequence, buf, row_names=configs['class_names'], col_names=np.array(IMAGENETCLASSES))
            buf.seek(0)
            im = transforms.ToTensor()(Image.open(buf))
        logger.add_image("mapping-matrix", im, epoch)
        logger.add_scalar("test/acc", acc, epoch)

        # Save CKPT
        state_dict = {
            "adv_program_dict": adv_program.state_dict(),
            "optimizer_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "mapping_sequence": mapping_sequence,
            "mapping_method": mapping_method,
            "mapping_num": mapping_num,
        }
        if acc > best_acc:
            best_acc = acc
            state_dict['best_acc'] = best_acc
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))
        torch.save(state_dict, os.path.join(save_path, 'ckpt.pth'))
    