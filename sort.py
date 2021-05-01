import os
import glob
import time

import torch
import torch.nn as nn
from shutil import copyfile, move

from model import GarbageModel
from util import pil_loader, prepare_image


class EnsembleModel(nn.Module):
    def __init__(self, models, input_size, classes):
        super(EnsembleModel, self).__init__()
        self.models = models
        self.input_size = input_size
        self.classes = classes

    def forward(self, x):
        predictions = [m(x) for m in self.models]
        return torch.mean(torch.stack(predictions), dim=0)


def make_ensemble(paths, device):
    print(" * Loading ensemble ...")
    # Load ensemble
    emodels = []
    for i in range(len(paths)):
        m = GarbageModel.load_from_checkpoint(paths[i], map_location=device)
        m.eval()
        m.freeze()
        if i==0:
            print("Categories :", m.hparams.classes)
            classes = m.hparams.classes
            input_size = m.hparams.input_size
        if classes == m.hparams.classes and input_size == m.hparams.input_size:
            print("Adding {}".format(paths[i]))
            m.to(device)
            emodels.append(m.model)
        else:
            print("Categories do not match : {}".format(paths[i]))
    print("Input Size :", input_size)
    model = EnsembleModel(emodels, input_size, classes)
    print(" * Ensemble loaded.")
    return model


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
    start_time = time.time()
    for i in range(max_count):
        img_color = pil_loader(images[i])
        img = prepare_image(img_color, model.input_size).to(device)
        yclass = model(img)
        class_prob, class_num = torch.max(yclass, dim=1)
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
    # print("Distribution of Labels:", [x / max_count for x in counts])
    duration = time.time() - start_time
    print(" * Sort Complete")
    print(" * Duration {:.2f} Seconds".format(duration))
    print(" * {:.2f} Images per Second".format(max_count/duration))


if __name__ == "__main__":

    root = "data/unsorted"
    num = 1000

    model_paths = [
        "lightning_logs/version_24/checkpoints/epoch=19-step=599.ckpt",
        ]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device : {}".format(device))

    model = make_ensemble(model_paths, device)

    sort_folder(model, device, root, num)
