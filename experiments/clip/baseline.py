from tqdm import tqdm
import argparse
import os
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import clip

import sys
sys.path.append(".")
from datatools.prepare_data import prepare_clip_data
from tools.misc import *
from tools.gen_text_embedding import get_saparate_text_embedding
from models.adv_program import VisualPrompt
from cfg import *


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--dataset', choices=["cifar10", "cifar100", "abide", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], default='cifar10')
    p.add_argument('--template-number', type=int, default=0)
    p.add_argument('--epoch', type=int, default=200)
    p.add_argument('--lr', type=float, default=40)
    args = p.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)

    exp = f"visual-prompt-clip-baseline"
    save_path = os.path.join(results_path, exp, gen_folder_name(args))

    TEMPLATES = [DEFAULT_TEMPLATE] + ENSEMBLE_TEMPLATES
    model, preprocess = clip.load("ViT-B/32")
    convert_models_to_fp32(model)
    model.eval()
    model.requires_grad_(False)
    loaders, class_names = prepare_clip_data(dataset=args.dataset, data_path=data_path, preprocess=preprocess)
    txt_emb = get_saparate_text_embedding(class_names, TEMPLATES[args.template_number], model)
    def network(x):
        x_emb = model.encode_image(x)
        x_emb /= x_emb.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * x_emb @ txt_emb.t()
        return logits

    visual_prompt = VisualPrompt(224, 30).to(device)

    # Optimizer
    optimizer = torch.optim.SGD(visual_prompt.parameters(), lr=args.lr, momentum=0.9)
    t_max = args.epoch * len(loaders['train'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    # Make Dir
    os.makedirs(save_path, exist_ok=True)
    logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    # Train
    best_acc = 0. 
    scaler = GradScaler()
    for epoch in range(args.epoch):
        visual_prompt.train()
        total_num = 0
        true_num = 0
        loss_sum = 0
        pbar = tqdm(loaders['train'], total=len(loaders['train']),
                desc=f"Epo {epoch} Training Lr {optimizer.param_groups[0]['lr']:.1e}", ncols=100)
        for i, (x, y) in enumerate(pbar):
            pbar.set_description_str(f"Epo {epoch} Training Lr {optimizer.param_groups[0]['lr']:.1e}", refresh=True)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                fx = network(visual_prompt(x))
                loss = F.cross_entropy(fx, y, reduction='mean')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            loss_sum += loss.item() * fx.size(0)
            pbar.set_postfix_str(f"Acc {100*true_num/total_num:.2f}%")
            scheduler.step()
        logger.add_scalar("train/acc", true_num/total_num, epoch)
        logger.add_scalar("train/loss", loss_sum/total_num, epoch)

        # Test
        visual_prompt.eval()
        total_num = 0
        true_num = 0
        pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Epo {epoch} Testing", ncols=100)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                fx = network(visual_prompt(x))
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            acc = true_num/total_num
            pbar.set_postfix_str(f"Acc {100*acc:.2f}%")
        logger.add_scalar("test/acc", acc, epoch)

        # Save CKPT
        state_dict = {
            "visual_prompt_dict": visual_prompt.state_dict(),
            "optimizer_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        }
        if acc > best_acc:
            best_acc = acc
            state_dict['best_acc'] = best_acc
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))
        torch.save(state_dict, os.path.join(save_path, 'ckpt.pth'))