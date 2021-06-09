import os
import argparse

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import pytorch_lightning as pl

from model import GarbageModel
from imbalance import ImbalancedSampler
from util import AddGaussianNoise


def get_args():
    parser = argparse.ArgumentParser()
    # Init and setup
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--ngpu', default=1, type=int)
    parser.add_argument('--benchmark', default=False, action='store_true')
    parser.add_argument('--imbalance', default=False, action='store_true')
    parser.add_argument('--mean', nargs=3, default=[0.485, 0.456, 0.406], type=float)
    parser.add_argument('--std', nargs=3, default=[0.229, 0.224, 0.225], type=float)
    parser.add_argument('--cutmix', default=0, type=float)
    # Model parameters
    parser.add_argument('--model', default='shufflenet_v2_x1_0', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    # Training Hyperparamters
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--split', default=0.1, type=float)
    parser.add_argument('--precision', default=32, type=int, choices=[16, 32])
    # Optimizer
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--opt', default='adam', type=str, choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr', default=4e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--nesterov', default=False, action='store_true')
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--accumulate', default=1, type=int)
    # Scheduler
    parser.add_argument('--scheduler', default=None, type=str, choices=['step', 'plateau', 'exp'])
    parser.add_argument('--lr_gamma', default=0.2, type=float)
    parser.add_argument('--milestones', nargs='+', default=[10, 15], type=int)
    parser.add_argument('--patience', default=20, type=int)
    args = parser.parse_args()
    return args


def main(args):
    pl.seed_everything(args.seed)

    # Setup transforms
    valid_transform = T.Compose([
        T.Resize(args.input_size),
        T.CenterCrop(args.input_size),
        T.ToTensor(),
    ])
    train_transform = T.Compose([
        T.RandomResizedCrop(args.input_size, scale=(0.08, 1.0)),
        T.RandomChoice([
            T.RandomPerspective(distortion_scale=0.5, p=1),
            T.RandomAffine(degrees=10, shear=15),
            T.RandomRotation(degrees=30)
        ]),
        T.ColorJitter(brightness=0.16, contrast=0.15, saturation=0.5, hue=0.04),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomGrayscale(),
        T.ToTensor(),
        AddGaussianNoise(std=0.01)
    ])

    # Get the datasets
    train_ds = ImageFolder(root=args.root, transform=train_transform)
    valid_ds = ImageFolder(root=args.root, transform=valid_transform)
    args.classes = train_ds.classes

    # Stratify the split by class
    train_idx, val_idx = train_test_split(list(range(len(train_ds))), test_size=args.split, random_state=args.seed , stratify=train_ds.targets)
    print("Split: Total={}, Train={}, Val={}.".format(len(train_ds), len(train_idx), len(val_idx)))

    # Get Dataset and Dataloaders
    if args.imbalance:
        train_sampler = ImbalancedSampler(train_ds, train_idx)
    else:
        train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # Check if the batches are balanced-ish
    # counts = [0]*len(train_ds.classes)
    # for j in range(10):
    #     for i in train_sampler:
    #         l = train_ds.targets[i]
    #         counts[l] += 1
    # print("Counts :", counts)

    train_loader = DataLoader(dataset=train_ds, batch_size=args.batch, sampler=train_sampler,
            num_workers=args.workers, persistent_workers=(True if args.workers > 0 else False),
            pin_memory=True)
    val_loader = DataLoader(dataset=valid_ds, batch_size=args.batch, sampler=val_sampler,
            num_workers=args.workers, persistent_workers=(True if args.workers > 0 else False),
            pin_memory=True)

    # x, y = next(iter(train_loader))
    # print(type(x), x.device, x.type, x.shape)
    # print(type(y), y.device, y.dtype, y.shape)

    model = GarbageModel(**vars(args))
    print(model.hparams)

    trainer = pl.Trainer(
        accumulate_grad_batches=args.accumulate,
        benchmark=args.benchmark,  # cudnn.benchmark
        deterministic=True,  # cudnn.deterministic
        gpus=args.ngpu,
        precision=args.precision,
        progress_bar_refresh_rate=10,
        max_epochs=args.epochs,
    )

    trainer.fit(
        model=model,
        train_dataloader=train_loader,
        val_dataloaders=val_loader
    )


if __name__ == "__main__":
    args = get_args()

    main(args)
