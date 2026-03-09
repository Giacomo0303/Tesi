import os, shutil
from timm.data import create_dataset
from torchvision import transforms
import math
from src.Datasets.Dataset import BaseDataset
from torch.utils.data import DataLoader


class ImageNet(BaseDataset):
    def __init__(self, root_path, batch_size, model_name, seed):
        super().__init__(root_path=root_path, img_size=224, batch_size=batch_size, mean_std="imagenet",
                         model_name=model_name, seed=seed)
        self.model_name = model_name
        self.correct_val_structure()
        self.class_dict = self.get_classes_dict()

        self.train_dataset = create_dataset(name="", root=root_path, split="train", is_training=True)
        self.test_dataset = create_dataset(name="", root=root_path, split="val", is_training=False)
        self.train_dataset.transform = self.get_transform()
        self.test_dataset.transform = self.get_transform(train=False)

    def get_transform(self, train=True):
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(
                    size=self.img_size,
                    scale=(0.8, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            crop_pct = 0.9
            scale_size = int(math.floor(self.img_size / crop_pct))  # 224 / 0.9 = 248

            return transforms.Compose([
                transforms.Resize(scale_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])

    def get_train_loader(self, num_workers):
        pass

    def get_test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_val_loader(self):
        return None

    def get_search_loader(self, n_per_classes):
        pass

    def get_classes_dict(self):
        folder_idx = self.train_dataset.parser.class_to_idx
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





