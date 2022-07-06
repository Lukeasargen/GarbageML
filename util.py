import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image


def time_to_string(t):
    if t > 3600: return "{:.2f} hours".format(t/3600)
    if t > 60: return "{:.2f} minutes".format(t/60)
    else: return "{:.2f} seconds".format(t)


""" https://github.com/NVIDIA/DeepLearningExamples/blob/8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/ConvNets/image_classification/smoothing.py """
class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, std=0.01):
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std * np.random.uniform()

    def __repr__(self):
        return self.__class__.__name__ + f'(std={self.std})'

def pil_loader(path):
    return Image.open(open(path, 'rb')).convert('RGB')


def prepare_image(img, size=None, centercrop=False):
    if size:
        img = T.Resize(size, interpolation=T.InterpolationMode.BICUBIC)(img)
        if centercrop:
            img = T.CenterCrop(size)(img)
    return T.ToTensor()(img).unsqueeze_(0)


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def crop_max_square(pil_img, size):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size)).resize((size, size))


class EnsembleModel(nn.Module):
    def __init__(self, models, input_size, classes):
        super(EnsembleModel, self).__init__()
        self.models = models
        self.input_size = input_size
        self.classes = classes

    def forward(self, x):
        predictions = [torch.softmax(m(x.clone()), dim=1) for m in self.models]
        return torch.mean(torch.stack(predictions), dim=0)


def make_ensemble(paths, plmodel, device):
    print(" * Loading ensemble ...")
    # Load ensemble
    emodels = []
    for i in range(len(paths)):
        m = plmodel.load_from_checkpoint(paths[i], map_location=device)
        m.eval()
        m.freeze()
        if i==0:  # The first model sets the inputs for the rest
            classes = m.hparams.classes
            input_size = m.hparams.input_size
        if classes == m.hparams.classes:
            print("Adding {}".format(paths[i]))
            m.to(device)
            emodels.append(m)
        else:
            print("Did not add : {}".format(paths[i]))
    print("Categories :", classes)
    print("Input Size :", input_size)
    model = EnsembleModel(emodels, input_size, classes)
    print(" * Ensemble loaded.")
    return model
