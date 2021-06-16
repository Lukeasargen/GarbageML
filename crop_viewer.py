
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def main():
    root = r"C:\Users\lukeasargen\projects\data\test512"
    batch_size = 4
    input_size = 256

    nrow = 5
    valid_transform = T.Compose([
        T.Resize(int(1.2*input_size)),
        T.FiveCrop(input_size),
        # T.TenCrop(input_size),
        T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])),
    ])

    valid_ds = ImageFolder(root=root, transform=valid_transform)
    val_loader = DataLoader(dataset=valid_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    print("{} Test Samples.".format(len(valid_ds)))

    data, target = next(iter(val_loader))

    bs, ncrops, c, h, w = data.size()
    data = data.view(-1, c, h, w)

    grid_img = make_grid(data, nrow=nrow)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()



if __name__ == "__main__":
    main()

