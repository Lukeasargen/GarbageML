import os
import time

from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from multiprocessing import pool
from multiprocessing.dummy import Pool as ThreadPool

from util import pil_loader

in_root = r"C:\Users\LUKE_SARGEN\projects\classifier\data\subset"
out_root = r"C:\Users\LUKE_SARGEN\projects\classifier\data\subset320"
out_size = int(320)
threads = 4

train_ds = ImageFolder(root=in_root)

# Make output folders
if not os.path.exists(out_root):
    os.mkdir(out_root)
out_folders = []  # Path to destination folder
for c in train_ds.classes:
    c_path = os.path.join(out_root, c)
    out_folders.append(c_path)
    if not os.path.exists(c_path):
        os.mkdir(c_path)


def resize_save(idx):
    path, target = train_ds.samples[idx]
    name = os.path.split(path)[1]  # get the file path, split, get filename w extension
    out = os.path.join(out_root, train_ds.classes[target], name)
    if not os.path.exists(out):
        img = pil_loader(path)
        x = T.Resize(out_size)(img)
        x.save(out)


start_time = time.time()
print(" * Starting resize...")

pool = ThreadPool(threads)
pool.map(resize_save, list(range(len(train_ds))))

duration = time.time() - start_time
print(" * Resize Complete")
print(" * Duration {:.2f} Seconds".format(duration))
print(" * {:.2f} Images per Second".format(len(train_ds)/duration))

