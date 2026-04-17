import copy
import os, shutil
import numpy as np
from torchvision import transforms
import math
from src.Datasets.Dataset import BaseDataset
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from timm.data import create_transform


class ImageNet(BaseDataset):
    def __init__(self, root_path, batch_size, model_name, seed=42, train_size=0.95):
        super().__init__(root_path=root_path, img_size=224, batch_size=batch_size, mean_std="imagenet",
                         model_name=model_name, seed=seed)
        self.model_name = model_name
        self.correct_val_structure()
        self.train_set, self.val_set, self.test_set, self.search_set = self.split_dataset(train_size=train_size)
        self.classes = self.train_set.dataset.classes
        self.num_classes = len(self.classes)
        self.class_dict = self.get_classes_dict()

    def get_transform(self, train=True):
        if train:
            return create_transform(
                input_size=self.img_size,
                is_training=True,
                color_jitter=0.4,
                auto_augment='rand-m9-mstd0.5-inc1',  # La policy RandAugment esatta del paper
                interpolation='bicubic',  # I ViT amano la bicubica, non la bilineare
                re_prob=0.0,  # 0.0 = Niente Random Erasing (come da indicazioni di Touvron)
                mean=self.mean,
                std=self.std
            )
        else:
            crop_pct = 0.9
            scale_size = int(math.floor(self.img_size / crop_pct))  # 224 / 0.9 = 248

            return transforms.Compose([
                transforms.Resize(scale_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])

    def split_dataset(self, train_size):
        train_set = ImageFolder(root=os.path.join(self.root_path, "train"), transform=self.get_transform())
        test_set = ImageFolder(root=os.path.join(self.root_path, "val"), transform=self.get_transform(train=False))
        targets = train_set.targets
        indices = np.arange(len(targets))

        train_indices, val_indices = train_test_split(
            indices,
            train_size=train_size,
            stratify=targets,
            random_state=self.seed
        )

        val_set = copy.copy(train_set)
        val_set.transform = self.get_transform(train=False)

        search_set = copy.copy(train_set)
        search_set.transform = self.get_transform(train=True)

        train_set = Subset(train_set, train_indices)
        val_set = Subset(val_set, val_indices)
        search_set = Subset(search_set, train_indices)

        return train_set, val_set, test_set, search_set

    def get_train_loader(self, num_workers):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=num_workers,
                          pin_memory=True, drop_last=True)

    def get_val_loader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def get_test_loader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

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

        gen.shuffle(final_indices)

        return DataLoader(Subset(self.search_set, final_indices), batch_size=self.batch_size, shuffle=False,
                          num_workers=3, pin_memory=True)

    def get_classes_dict(self):
        folder_idx = self.train_set.dataset.class_to_idx
        idx_class = {}
        with open(os.path.join(self.root_path, "LOC_synset_mapping.txt"), "r") as f:
            for line in f:
                folder, cls = line.split(" ")[0], line.split(" ", 1)[1].split(",")[0].strip()
                if folder in folder_idx:
                    idx_class[folder_idx[folder]] = cls

        return idx_class

    def correct_val_structure(self):
        val_path = os.path.join(self.root_path, "val")
        files = os.listdir(val_path)
        jpegs = [f for f in files if f.endswith(".JPEG")]
        if len(jpegs) == 0:
            return

        mapping = {}
        with open(os.path.join(self.root_path, "LOC_val_solution.csv"), "r") as f:
            next(f)
            for line in f:
                file, folder = line.split(",")[0], line.split(",")[1].split(" ")[0]
                mapping[file + ".JPEG"] = folder

        for jpeg in jpegs:
            os.makedirs(os.path.join(val_path, mapping[jpeg]), exist_ok=True)
            shutil.move(os.path.join(val_path, jpeg), os.path.join(val_path, mapping[jpeg], jpeg))
