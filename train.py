import argparse
from collections import Counter
import os

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from model import GarbageModel
from imbalance import ImbalancedSampler
from util import AddGaussianNoise


def get_args():
    # For plateau, early loss, checkpointing
    metrics_choices = ["loss", "accuracy", "average_precision", "average_recall", "weighted_f1"]

    parser = argparse.ArgumentParser()
    # Init and setup
    parser.add_argument('--seed', type=int)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--name', default='default', type=str)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--ngpu', default=1, type=int)
    parser.add_argument('--benchmark', default=False, action='store_true')
    parser.add_argument('--precision', default=32, type=int, choices=[16, 32])
    # Dataset
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--accumulate', default=1, type=int)
    parser.add_argument('--split', default=0.1, type=float)
    parser.add_argument('--imbalance_oversample', default=False, action='store_true')
    parser.add_argument('--imbalance_weights', default=False, action='store_true')
    parser.add_argument('--mean', nargs=3, default=[0.485, 0.456, 0.406], type=float)
    parser.add_argument('--std', nargs=3, default=[0.229, 0.224, 0.225], type=float)
    # Model parameters
    parser.add_argument('--model', default='shufflenet_v2_x0_5', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--finetune_lr', default=1e-6, type=float)
    parser.add_argument('--finetune_after', default=-1, type=float)
    parser.add_argument('--dropout', default=0.0, type=float)
    # Attention classifier, add .attn to model name
    parser.add_argument('--attn_embd', default=512, type=int)
    parser.add_argument('--attn_dim', default=32, type=int)
    parser.add_argument('--attn_heads', default=6, type=int)
    parser.add_argument('--attn_layers', default=1, type=int)
    parser.add_argument('--attn_ff_multi', default=2, type=int)
    parser.add_argument('--attn_pos_size', default=2, type=int)
    parser.add_argument('--attn_avg_tokens', default=False, action='store_true')
    # Optimizer
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--max_steps', default=None, type=int)
    parser.add_argument('--opt', default='adam', type=str, choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr', default=4e-3, type=float)
    parser.add_argument('--lr_warmup_steps', default=0, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--nesterov', default=False, action='store_true')
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--grad_clip', default='norm', type=str, choices=['value', 'norm'])
    parser.add_argument('--clip_value', default=0, type=float)    
# Scheduler
    parser.add_argument('--scheduler', default=None, type=str, choices=['step', 'plateau', 'exp'])
    parser.add_argument('--lr_gamma', default=0.2, type=float)
    parser.add_argument('--milestones', nargs='+', default=[10, 15], type=int)
    parser.add_argument('--plateau_patience', default=20, type=int)
    parser.add_argument('--plateau_monitor', default='loss', type=str, choices=metrics_choices)
    # Callbacks
    parser.add_argument('--val_interval', default=1, type=int)
    parser.add_argument('--val_percent', default=1.0, type=float)
    parser.add_argument('--save_top_k', default=1, type=int)
    parser.add_argument('--save_monitor', default='accuracy', type=str, choices=metrics_choices)
    parser.add_argument('--early_stop', default=None, type=str, choices=metrics_choices)
    parser.add_argument('--early_stop_patience', default=20, type=int)
    # Augmentations
    parser.add_argument('--cutmix', default=0, type=float)
    parser.add_argument('--cutmix_prob', default=0.5, type=float)
    parser.add_argument('--aug_scale', default=0.08, type=float)
    parser.add_argument('--label_smoothing', default=0.0, type=float)
    args = parser.parse_args()
    return args


def main(args):
    pl.seed_everything(args.seed)

    # Increment to find the next availble name
    logger = TensorBoardLogger(save_dir="logs", name=args.name)    
    dirpath = f"logs/{args.name}/version_{logger.version}"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
    callbacks = [
        ModelCheckpoint(
            monitor=args.save_monitor+"/val_epoch",
            dirpath=dirpath,
            filename="topk_{step}",
            save_top_k=args.save_top_k,
            mode='min' if args.save_monitor=='loss' else 'max',
            period=1,  # Check every validation epoch
            save_last=True,
            save_on_train_epoch_end=False,
        )
    ]

    if args.early_stop is not None:
        callbacks.append(
            EarlyStopping(
                monitor=args.early_stop+"/val_epoch",
                patience=args.early_stop_patience,
                mode='min' if args.early_stop=='loss' else 'max',
            )
        )

    # Setup transforms
    valid_transform = T.Compose([
        T.Resize(args.input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(args.input_size),
        T.ToTensor(),
    ])
    train_transform = T.Compose([
        T.RandomResizedCrop(args.input_size, scale=(args.aug_scale, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        # T.RandomChoice([
        #     T.RandomPerspective(distortion_scale=0.2, p=1),
        #     T.RandomAffine(degrees=10, shear=15),
        #     T.RandomRotation(degrees=15)
        # ]),
        T.RandomPerspective(distortion_scale=0.2, p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.5, hue=0.1),
        T.RandomHorizontalFlip(),
        # T.RandomVerticalFlip(),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        AddGaussianNoise(std=0.02)
    ])

    # Get the datasets
    train_ds = ImageFolder(root=args.root, transform=train_transform)
    valid_ds = ImageFolder(root=args.root, transform=valid_transform)
    args.classes = train_ds.classes

    # Stratify the split by class
    train_idx, val_idx = train_test_split(list(range(len(train_ds))), test_size=args.split, random_state=args.seed , stratify=train_ds.targets)
    print("Split: Total={}, Train={}, Val={}.".format(len(train_ds), len(train_idx), len(val_idx)))

    # Get Dataset and Dataloaders
    if args.imbalance_oversample:
        train_sampler = ImbalancedSampler(train_ds, train_idx)
    else:
        train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # Check if the batches are balanced-ish
    counts = [0]*len(train_ds.classes)
    for j in range(10):
        for i in train_sampler:
            l = train_ds.targets[i]
            counts[l] += 1
    print("Sampling Counts :", counts)

    class_weights = torch.Tensor(list(Counter(train_ds.targets).values()))
    class_weights = sum(class_weights)/(len(class_weights)*class_weights)
    args.class_weights = class_weights
    print("Class Weights :", class_weights)

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

    trainer = pl.Trainer(
        accumulate_grad_batches=args.accumulate,
        benchmark=args.benchmark,  # cudnn.benchmark
        callbacks=callbacks,
        check_val_every_n_epoch=args.val_interval,
        deterministic=True,  # cudnn.deterministic
        gpus=args.ngpu,
        gradient_clip_algorithm=args.grad_clip,
        gradient_clip_val=args.clip_value,
        logger=logger,
        precision=args.precision,
        progress_bar_refresh_rate=1,
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        num_sanity_val_steps=0,
        limit_val_batches=args.val_percent,
        log_every_n_steps=1,
    )

    trainer.fit(
        model=model,
        train_dataloader=train_loader,
        val_dataloaders=val_loader
    )


if __name__ == "__main__":
    args = get_args()

    main(args)
