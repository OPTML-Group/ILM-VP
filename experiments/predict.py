from functools import partial
import os
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from tqdm import tqdm
import argparse

import sys
sys.path.append(".")
from datatools.prepare_data import prepare_ar_resnet_data
from datatools.const import IMAGENETNORMALIZE
from models.adv_program import AdvProgram
from cfg import *


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ap-path', type=str, required=True)
    p.add_argument('--dataset', choices=["cifar10", "cifar100", "abide", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], default='cifar10')
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

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

    save_path = args.ap_path
    loaders, configs = prepare_ar_resnet_data(args.dataset, data_path=data_path)
    normalize = transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std'])

    # Network
    network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    network.requires_grad_(False)
    network = network.to(device)
    network.eval()

    # Adversarial Program
    adv_program = AdvProgram(224, mask=configs['mask'], normalize=normalize).to(device)
    state_dict = torch.load(os.path.join(args.ap_path, "best.pth"))
    adv_program.load_state_dict(state_dict["adv_program_dict"])
    mapping_sequence = state_dict["mapping_sequence"]
    label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
        
    # Test
    adv_program.eval()
    total_num = 0
    true_num = 0
    pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Testing", ncols=100)
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
    torch.save(
        {
            'fx0s': fx0s,
            'ys': ys,
        },
        os.path.join(args.ap_path, "best_prediction.pth")
    )