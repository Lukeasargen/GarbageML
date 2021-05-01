import os
import time

from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from multiprocessing import pool
from multiprocessing.dummy import Pool as ThreadPool


in_root = "data/full"
out_root = "data/full448"
out_size = int(448)
threads = 6

resize = T.Resize(out_size)

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
    x, y = train_ds[idx]
    name = os.path.split(train_ds.samples[idx][0])[1]  # get the file path, split, get filename w extension
    x = resize(x)
    out = os.path.join(out_root, train_ds.classes[y], name)
    x.save(out)


start_time = time.time()
print(" * Starting resize...")

pool = ThreadPool(threads)
pool.map(resize_save, list(range(len(train_ds))))

duration = time.time() - start_time
print(" * Resize Complete")
print(" * Duration {:.2f} Seconds".format(duration))
print(" * {:.2f} Images per Second".format(len(train_ds)/duration))

