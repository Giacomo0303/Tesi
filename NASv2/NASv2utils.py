import numpy as np
from torch.utils.data import Subset

def get_search_set(train_set, n_per_classes, n_classes):
    train_indices = np.arange(len(train_set))

    targets = np.array(train_set.dataset.targets)
    train_labels = targets[train_set.indices]

    gen = np.random.default_rng()
    final_indices = []

    for cls in range(n_classes):
        cls_indices = train_indices[train_labels == cls]

        if len(cls_indices) >= n_per_classes:
            selected_indices = gen.choice(cls_indices, size=n_per_classes, replace=False)
        else:
            selected_indices = cls_indices

        final_indices.extend(selected_indices.tolist())

    return Subset(train_set, final_indices)
