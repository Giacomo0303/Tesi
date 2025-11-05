import os
from random import shuffle
import cv2
from tqdm import tqdm
from shutil import rmtree

train_path = "/Places365Simplified/train"
set1_path = "/Sets/Set1"
set2_path = "/Sets/Set2"

train_val_split = 0.95

rmtree("/Sets")

#creo la cartella dei due set
os.makedirs(set1_path, exist_ok=True)
os.makedirs(set2_path, exist_ok=True)
os.makedirs(os.path.join(set1_path, "train"), exist_ok=True)
os.makedirs(os.path.join(set2_path, "train"), exist_ok=True)
os.makedirs(os.path.join(set1_path, "val"), exist_ok=True)
os.makedirs(os.path.join(set2_path, "val"), exist_ok=True)

class_list = os.listdir(train_path)

for cl in tqdm(class_list, desc="classi"):
    os.makedirs(os.path.join(set1_path, "train", cl), exist_ok=True)
    os.makedirs(os.path.join(set1_path, "val", cl), exist_ok=True)
    os.makedirs(os.path.join(set2_path, "train", cl), exist_ok=True)
    os.makedirs(os.path.join(set2_path, "val", cl), exist_ok=True)

    images = os.listdir(os.path.join(train_path, cl))
    shuffle(images)
    imgs_sets = [images[:int(len(images)/2)], images[int(len(images)/2):]]

    for i, set_i in enumerate(imgs_sets):

        train_set = set_i[:int(train_val_split * len(set_i))]
        val_set = set_i[int(train_val_split * len(set_i)):]
        dest_path = set1_path if i == 0 else set2_path

        for img_name in train_set:
            img = cv2.imread(os.path.join(train_path, cl, img_name))
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(dest_path , "train", cl, img_name), img)
        
        for img_name in val_set:
            img = cv2.imread(os.path.join(train_path, cl, img_name))
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(dest_path , "val", cl, img_name), img)

            






