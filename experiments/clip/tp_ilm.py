from tqdm import tqdm
import argparse
from functools import partial
import os
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import clip
from PIL import Image
import io

import sys
sys.path.append(".")
from datatools.prepare_data import prepare_clip_data
from tools.frequency_mapping import generate_label_mapping_by_frequency, get_dist_matrix
from tools.misc import *
from tools.gen_text_embedding import get_saparate_text_embedding
from tools.draw_mapping import plot_mapping
from models.adv_program import VisualPrompt
from cfg import *


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--dataset', choices=["cifar10", "cifar100", "abide", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], default='cifar10')
    p.add_argument('--mapping-num', type=int, default=1)
    p.add_argument('--mapping-method', type=str, choices=['mean', 'max', 'solo'], default='solo')
    p.add_argument('--mapping-interval', type=int, required=True)

    p.add_argument('--epoch', type=int, default=200)
    p.add_argument('--lr', type=float, default=40)
    args = p.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)

    assert args.mapping_method in ["solo", "mean", "max"]
    if args.mapping_method == 'solo':
        args.mapping_num = 1
    def label_mapping_base(logits, mapping_sequence):
        if args.mapping_method == "mean":
            modified_logits = logits[:, mapping_sequence].reshape(logits.size(0), -1, args.mapping_num).mean(-1)
        elif args.mapping_method == "max":
            modified_logits = logits[:, mapping_sequence].reshape(logits.size(0), -1, args.mapping_num).max(-1).values
        else:
            modified_logits = logits[:, mapping_sequence]
        return modified_logits

    exp = f"ar-prompt-mapping-clip"
    save_path = os.path.join(results_path, exp, gen_folder_name(args))

    model, preprocess = clip.load("ViT-B/32")
    convert_models_to_fp32(model)
    model.eval()
    model.requires_grad_(False)
    loaders, class_names = prepare_clip_data(dataset=args.dataset, data_path=data_path, preprocess=preprocess)
    templates = [DEFAULT_TEMPLATE]+ENSEMBLE_TEMPLATES
    txt_emb = torch.cat(get_saparate_text_embedding(class_names, templates, model))
    emb_names = np.array([f"T{i//len(class_names)} {class_names[i%len(class_names)]}" for i in range(txt_emb.size(0))])
    def network(x):
        x_emb = model.encode_image(x)
        x_emb /= x_emb.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * x_emb @ txt_emb.t()
        return logits
    mapping_network = network

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
        if epoch % args.mapping_interval == 0:
            mapping_sequence = generate_label_mapping_by_frequency(visual_prompt, mapping_network, loaders['train'], args.mapping_num)
            label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
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
                fx = label_mapping(network(visual_prompt(x)))
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
        fx0s = []
        ys = []
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            ys.append(y)
            with torch.no_grad():
                fx0 = network(visual_prompt(x))
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
            plot_mapping(mapping_matrix, mapping_sequence, buf, row_names=class_names, col_names=emb_names)
            buf.seek(0)
            im = transforms.ToTensor()(Image.open(buf))
            logger.add_image("mapping-matrix", im, epoch)
        logger.add_scalar("test/acc", acc, epoch)

        # Save CKPT
        state_dict = {
            "visual_prompt_dict": visual_prompt.state_dict(),
            "optimizer_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "mapping_sequence": mapping_sequence,
            "mapping_method": args.mapping_method,
            "mapping_num": args.mapping_num,
        }
        if acc > best_acc:
            best_acc = acc
            state_dict['best_acc'] = best_acc
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))
        torch.save(state_dict, os.path.join(save_path, 'ckpt.pth'))