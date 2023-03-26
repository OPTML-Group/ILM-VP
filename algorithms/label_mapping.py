
def label_mapping_base(logits, mapping_sequence):
    modified_logits = logits[:, mapping_sequence]
    return modified_logits