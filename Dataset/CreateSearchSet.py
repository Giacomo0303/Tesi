import os
from shutil import copyfile
from random import shuffle
from tqdm import tqdm
from shutil import rmtree

path = "D:\\Tesi\\Sets\\Set1"
classes = os.listdir(os.path.join(path, "val"))
rmtree(os.path.join(path, "search"))
os.makedirs(os.path.join(path, "search"), exist_ok=True)
images_per_class = 10

for cl in tqdm(classes):
    os.makedirs(os.path.join(path, "search", cl), exist_ok=True)
    images = os.listdir(os.path.join(path, "val", cl))
    shuffle(images)
    images = images[:images_per_class]
    for img in images:
        copyfile(os.path.join(path,"val", cl, img), os.path.join(path, "search", cl, img))



