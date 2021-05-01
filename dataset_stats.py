import time

import numpy as np
from torchvision.datasets import ImageFolder

from util import pil_loader, prepare_image


root = "data/full448"
num = 10000

sample_ds = ImageFolder(root=root)

mean = 0.0
var = 0.0
n = min(len(sample_ds), num)  # Go through the whole dataset if possible
t0 = time.time()
t1 = t0
for i in range(n):
    # img in shape [W, H, C]
    img_path, y = sample_ds.samples[i]
    img = np.array(pil_loader(img_path)) / 255.0
    mean += np.mean(img, axis=(0, 1))
    var += np.var(img, axis=(0, 1))  # you can add var, not std
    if (i+1) % 100 == 0:
        t2 = time.time()
        print("{}/{} measured. Total time={:.2f}s. Images per second {:.2f}.".format(i+1, n, t2-t0, 100/(t2-t1)))
        t1 = t2
print("mean = [{:4.3f}, {:4.3f}, {:4.3f}]".format(*(mean/n)))
print("std = [{:4.3f}, {:4.3f}, {:4.3f}]".format(*np.sqrt(var/n)))
print("var :", var/n)

