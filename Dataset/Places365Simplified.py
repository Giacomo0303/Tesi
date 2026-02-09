import os
from shutil import rmtree
from Dataset.Dataset import BaseDataset


class Places365Simplified(BaseDataset):
    def __init__(self, root_path, img_size, batch_size, mean_std, model_name, seed):
        super().__init__(root_path, img_size, batch_size, mean_std, model_name, seed)
        # Esegue la pulizia e assegna le classi RIMASTE
        self.classes = self.simplify_dataset()
        self.num_classes = len(self.classes)

    def simplify_dataset(self):
        removed_path = os.path.join(self.root_path, "removed_class.txt")
        if not os.path.exists(removed_path):
            print("File removed_class.txt non trovato. Nessuna pulizia effettuata.")
            return sorted(os.listdir(os.path.join(self.root_path, "train")))

        with open(removed_path, "r") as f:
            removed_classes = [line.strip() for line in f.readlines()]

        # --- Semplificazione Train Set ---
        train_path = os.path.join(self.root_path, "train")
        train_classes_pre = os.listdir(train_path)

        for cl in train_classes_pre:
            if cl in removed_classes:
                rmtree(os.path.join(train_path, cl))

        final_train_classes = sorted(os.listdir(train_path))
        print(f"Train set: {len(train_classes_pre)} -> {len(final_train_classes)} classi.")

        # --- Semplificazione Val Set ---
        val_path = os.path.join(self.root_path, "val")
        val_classes_pre = os.listdir(val_path)

        for cl in val_classes_pre:
            if cl in removed_classes:
                rmtree(os.path.join(val_path, cl))

        final_val_classes = sorted(os.listdir(val_path))
        print(f"Val set: {len(val_classes_pre)} -> {len(final_val_classes)} classi.")

        # Check integrità
        if final_train_classes != final_val_classes:
            print(f"ATTENZIONE: Disallineamento classi Train/Val!")
            print(f"Differenza: {set(final_val_classes) ^ set(final_train_classes)}")
        else:
            print("Check integrità superato: Train e Val hanno le stesse classi.")

        return final_train_classes

    def get_transform(self, train=True):
        pass

    def get_train_loader(self, num_workers):
        pass

    def get_val_loader(self):
        pass

    def get_test_loader(self):
        pass

    def search_loader(self):
        pass
