import torch
from matplotlib import pyplot as plt

from .heatmap import heatmap, annotate_heatmap


def plot_mapping(mapping_matrix, mapping_sequence, buf, row_names, col_names):
    if mapping_sequence.size(0) > 10:
        mapping_sequence = mapping_sequence[:10]
        source_max = 10
        extra_cols = 0
    else:
        source_max = len(mapping_sequence)
        extra_cols = len(mapping_sequence)
    most_frequent_classes = torch.argsort(mapping_matrix.sum(-1), descending=True)[:extra_cols]
    showing_ind = mapping_sequence.tolist() + most_frequent_classes.tolist()
    mapping_matrix = (mapping_matrix * 100 / mapping_matrix.sum(0)).int()
    show_mat = mapping_matrix[showing_ind, :source_max]
    show_mat = show_mat.permute(1, 0)

    fig, ax = plt.subplots()
    im, cbar = heatmap(show_mat, row_names[:source_max], col_names[showing_ind], ax=ax,
                    cmap="YlGn", cbarlabel="Prediction (%)")
    texts = annotate_heatmap(im, valfmt="{x:d}")

    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close()