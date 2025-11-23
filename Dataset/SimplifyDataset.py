import os
from shutil import rmtree

base_path = "D:\\Tesi\\Places365Simplified"

#utilizzo le removed_class per assicurarmi che non siano presenti nel train e val principali

with open(os.path.join(base_path, "removed_class.txt"), "r") as f:
    removed_classes = f.readlines()

removed_classes = list(map(lambda x: x.rstrip(), removed_classes))

#semplificazione del train set

train_path = os.path.join(base_path, "train")
train_classes = os.listdir(train_path)
for cl in train_classes:
    if cl in removed_classes:
        rmtree(os.path.join(train_path, cl))
print(f"Classi originali train_set: {len(train_classes)}, dopo la rimozione: {len(os.listdir(train_path))}")

#semplificazione del val set

val_path = os.path.join(base_path, "val")
val_classes = os.listdir(val_path)
for cl in val_classes:
    if cl in removed_classes:
        rmtree(os.path.join(val_path, cl))
print(f"Classi originali train_set: {len(val_classes)}, dopo la rimozione: {len(os.listdir(val_path))}")

#check visivo finale

print(f"Differenza tra i due set: {set(val_classes) - set(train_classes)}")
print(f"Sono uguali: {set(val_classes) == set(train_classes)}")

    
