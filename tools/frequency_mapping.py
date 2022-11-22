import torch
from torch.nn.functional import one_hot
from tqdm import tqdm

def get_dist_matrix(fx, y):
    fx = one_hot(torch.argmax(fx, dim = -1), num_classes=fx.size(-1))
    dist_matrix = [fx[y==i].sum(0).unsqueeze(1) for i in range(len(y.unique()))]
    dist_matrix = torch.cat(dist_matrix, dim=1)
    return dist_matrix


def predictive_distribution_based_multi_label_mapping(dist_matrix, mlm_num: int):
    assert mlm_num * dist_matrix.size(1) <= dist_matrix.size(0), "source label number not enough for mapping"

    def check_mapping_matrix(mat):
        if mat.sum(1).max() > 1 or mat.sum(0).max() > mlm_num:
            return 0 # error
        elif (mat.sum(0) == mlm_num).all():
            return 1 # finished
        else:
            return 2

    mapping_matrix = torch.zeros_like(dist_matrix, dtype=int)
    dist_matrix_flat = dist_matrix.flatten() # same memory
    mapping_matrix_flat = mapping_matrix.flatten() # same memory

    while check_mapping_matrix(mapping_matrix) != 1:

        tmp_mapping_matrix = mapping_matrix.clone()
        tmp_mapping_matrix_flat = tmp_mapping_matrix.flatten() # same memory

        idx = dist_matrix_flat.argmax().item()
        tmp_mapping_matrix_flat[idx] = 1
        if dist_matrix_flat[idx] < 0:
            dist_matrix_flat[idx] -= 1
        else:
            dist_matrix_flat[idx] = -1

        if check_mapping_matrix(tmp_mapping_matrix) != 0:
            mapping_matrix_flat[idx] = 1

    return mapping_matrix


def predictive_distribution_based_multi_label_mapping_fast(dist_matrix, mlm_num: int):
    assert mlm_num * dist_matrix.size(1) <= dist_matrix.size(0), "source label number not enough for mapping"
    mapping_matrix = torch.zeros_like(dist_matrix, dtype=int)
    dist_matrix_flat = dist_matrix.flatten() # same memory
    for _ in range(mlm_num * dist_matrix.size(1)):
        loc = dist_matrix_flat.argmax().item()
        loc = [loc // dist_matrix.size(1), loc % dist_matrix.size(1)]
        mapping_matrix[loc[0], loc[1]] = 1
        dist_matrix[loc[0]] = -1
        if mapping_matrix[:, loc[1]].sum() == mlm_num:
            dist_matrix[:, loc[1]] = -1
    return mapping_matrix



def generate_label_mapping_by_frequency(adv_program, network, data_loader, mapping_num):
    device = next(adv_program.parameters()).device
    if hasattr(network, "eval"):
        network.eval()
    fx0s = []
    ys = []
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Frequency Label Mapping", ncols=100) if len(data_loader) > 20 else data_loader
    for x, y in pbar:
        if x.get_device() == -1:
            x, y = x.to(device), y.to(device)
        with torch.no_grad():
            fx0 = network(adv_program(x))
        fx0s.append(fx0)
        ys.append(y)
    fx0s = torch.cat(fx0s).cpu().float()
    ys = torch.cat(ys).cpu().int()
    if ys.size(0) != fx0s.size(0):
        assert fx0s.size(0) % ys.size(0) == 0
        ys = ys.repeat(int(fx0s.size(0) / ys.size(0)))
    dist_matrix = get_dist_matrix(fx0s, ys)
    pairs = torch.nonzero(predictive_distribution_based_multi_label_mapping_fast(dist_matrix, mapping_num))
    mapping_sequence = pairs[:, 0][torch.sort(pairs[:, 1]).indices.tolist()]
    return mapping_sequence


if __name__ == "__main__":
    a, b = torch.Tensor(397, 1000), torch.arange(397)
    m = get_dist_matrix(a, b)
    m1 = predictive_distribution_based_multi_label_mapping(m.clone(), 1)
    m2 = predictive_distribution_based_multi_label_mapping_fast(m.clone(), 1)
    print((m1 == m2).all())
    