import copy

import numpy as np
from torch.utils.data import random_split, Subset, DataLoader
from torchvision.datasets import CIFAR100
from torch import Generator
from Dataset.Dataset import BaseDataset
from torchvision import transforms


class Cifar100(BaseDataset):
    def __init__(self, root_path, img_size, batch_size, mean_std, model_name, train_size=0.9, seed=42):
        super().__init__(root_path, img_size, batch_size, mean_std, model_name, seed)
        self.train_set, self.val_set, self.test_set, self.search_set = self.split_dataset(train_size=train_size)
        self.classes = self.train_set.dataset.classes
        self.num_classes = len(self.classes)

    def get_transform(self, train=True):
        if train:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        return transforms.Compose(
            [transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor(),
             transforms.Normalize(mean=self.mean, std=self.std)])

    def split_dataset(self, train_size):
        train_set = CIFAR100(root=self.root_path, train=True, download=True, transform=self.get_transform(train=True))
        test_set = CIFAR100(root=self.root_path, train=False, download=True, transform=self.get_transform(train=False))

        total_size = len(train_set)
        train_size = int(train_size * total_size)
        val_size = total_size - train_size

        generator = Generator().manual_seed(self.seed)
        train_split, val_split = random_split(train_set, [train_size, val_size], generator=generator)

        val_set = copy.copy(train_set)
        val_set.transform = self.get_transform(train=False)

        search_set = copy.copy(train_set)
        search_set.transform = self.get_transform(train=False)

        train_set = Subset(train_set, train_split.indices)
        val_set = Subset(val_set, val_split.indices)
        search_set = Subset(search_set, train_split.indices)

        return train_set, val_set, test_set, search_set

    def get_train_loader(self, num_workers):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=num_workers,
                          pin_memory=True)

    def get_val_loader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def get_test_loader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

    def get_search_loader(self, n_per_classes):
        search_indices = np.arange(len(self.search_set))

        targets = np.array(self.search_set.dataset.targets)
        search_labels = targets[self.search_set.indices]

        gen = np.random.default_rng()
        final_indices = []

        for cls in range(self.num_classes):
            cls_indices = search_indices[search_labels == cls]

            if len(cls_indices) >= n_per_classes:
                selected_indices = gen.choice(cls_indices, size=n_per_classes, replace=False)
            else:
                selected_indices = cls_indices

            final_indices.extend(selected_indices.tolist())

        return DataLoader(Subset(self.search_set, final_indices), batch_size=self.batch_size, shuffle=False,
                          num_workers=1, pin_memory=True)
