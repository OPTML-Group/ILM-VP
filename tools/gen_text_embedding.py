import torch
from typing import List
from tqdm import tqdm
import clip

def get_text_ensemble_embedding(classnames, templates, model):
    device = next(model.parameters()).device
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, desc="Embedding texts", ncols=100):
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).to(device)
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

def get_saparate_text_embedding(classnames, templates, model):
    device = next(model.parameters()).device
    zeroshot_weights = []
    if isinstance(templates, List):
        for template in tqdm(templates, desc="Embedding texts", ncols=100):
            texts = [template.format(classname) for classname in classnames]
            texts = clip.tokenize(texts).to(device)
            with torch.no_grad():
                text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            zeroshot_weights.append(text_embeddings)
    else:
        texts = [templates.format(classname) for classname in classnames]
        texts = clip.tokenize(texts).to(device)
        with torch.no_grad():
            text_embeddings = model.encode_text(texts)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        zeroshot_weights = text_embeddings
    return zeroshot_weights