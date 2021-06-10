import os
import glob
import time

import torch
from torchvision import transforms as T
from shutil import copyfile, move

from model import GarbageModel
from util import pil_loader, prepare_image, make_ensemble


def sort_folder(model, device, root, num=None):
    print(" * Sorting folder : {} ...".format(root))
    # Create folders for categories
    class_folder_paths = []  # Absolute path to destination folder
    for cat in model.classes:
        cat_path = os.path.join(root, cat)
        class_folder_paths.append(cat_path)
        if not os.path.exists(cat_path):
            os.mkdir(cat_path)

    # Classify each image and cut-paste into label folder
    image_types = ["*.jpg", "*.png", "*.jpeg"]
    images = [f for ext in image_types for f in glob.glob(os.path.join(root, ext))]
    print("{} total images.".format(len(images)))
    max_count = min(num, len(images))
    print(" * Sorting {} ...".format(max_count))
    counts = [0]*len(model.classes)
    valid_transform = T.Compose([
        T.Resize(model.input_size),
        T.FiveCrop(model.input_size),
        # T.TenCrop(model.input_size),
        T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])),
    ])
    start_time = time.time()
    for i in range(max_count):
        img_color = pil_loader(images[i])

        # img = prepare_image(img_color, model.input_size).to(device)

        img = valid_transform(img_color).to(device)
        ncrops, c, h, w = img.size()
        img = img.view(-1, c, h, w)

        yclass = model(img)
        yclass = yclass.view(ncrops, -1).mean(0)

        class_prob, class_num = torch.max(yclass, dim=0)
        counts[int(class_num)] += 1
        try:
            move(images[i], class_folder_paths[int(class_num)])
        except:
            print("Failed to move {}".format(images[i]))

        if (i+1) % 50 == 0:
            count = i+1
            t2 = time.time() - start_time
            rate = count/t2
            est = t2/count * (max_count-count)
            print("{}/{} images. {:.2f} seconds. {:.2f} images per seconds. {:.2f} seconds remaining.".format(count, max_count, t2, rate, est))
    print("Labels per class :", counts)
    print("Distribution :", [c/sum(counts) for c in counts])
    duration = time.time() - start_time
    print(" * Sort Complete")
    print(" * Duration {:.2f} Seconds".format(duration))
    print(" * {:.2f} Images per Second".format(max_count/duration))


if __name__ == "__main__":

    root = "data/unsorted"
    num = 1000

    model_paths = [
        # "lightning_logs/version_2/checkpoints/epoch=199-step=11799.ckpt",
        "lightning_logs/version_3/checkpoints/epoch=199-step=5999.ckpt",
    ]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device : {}".format(device))

    model = make_ensemble(model_paths, GarbageModel, device)

    sort_folder(model, device, root, num)
