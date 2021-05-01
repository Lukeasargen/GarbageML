import torch
import torchvision.transforms as T
from PIL import Image


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, std=0.01):
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std
    
    def __repr__(self):
        return self.__class__.__name__ + '(std={1})'.format(self.std)


def pil_loader(path):
    return Image.open(open(path, 'rb')).convert('RGB')


def prepare_image(img, size=None):
    if size:
        img = crop_max_square(img, size)
    return T.ToTensor()(img).unsqueeze_(0)


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def crop_max_square(pil_img, size):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size)).resize((size, size))
