
This project was inspired by a friend who is competing in the Penn State Nittany AI Challenge.

https://nittanyai.psu.edu/alliance-programs/nittany-ai-challenge/

The goal is a model that can classify an image of a piece of waste into the correct bin.

https://sustainability.psu.edu/campus-efforts/operations/recycling-composting/

Also, this is my first project using pytorch lightning.

https://github.com/PyTorchLightning/pytorch-lightning

# Setup

Create a new conda environment
```
conda create --name pytorch
```

```
conda activate pytorch
```

Get pytorch installed. Command generated here: https://pytorch.org/
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
```

Requirements
```
pip install -r requirements.txt
```

Sometimes pytorch-lightning didn't install, so try these pip arguments.
```
pip install pytorch-lightning
pip install pytorch-lightning -U
pip install pytorch-lightning --ignore-installed dataclasses
```

Create the interactive notebook kernel:

```
conda install ipykernel jupyter
```

```
python -m ipykernel install --user --name pytorch --display-name "pytorch"
```

# Datasets

## TrashNet

I started with TrashNet (Yang and Thung, 2016) dataset. This is a pretty balanced dataset. The labels don't match the recycling bins, but I think it's decent starting point.

Dataset link: https://github.com/garythung/trashnet

Stats: 2527 images, 501 glass, 594 paper, 403 cardboard, 482 plastic, 410 metal, 137 trash

I downsampled the images to speed up the image loading pipeine. You can see how that is done in [resize.py](resize.py).

Sample of the dataset with labels and predictions (resnet18).

![trash_labels](/data/readme/trash_labels.png)


# Model

You can use any torchvision model. Just get the name string correct. Here is the code that creates the torchvision models.
```
from torchvision import models
from torchvision.transforms import Normalize

def get_model(args):
    # args.model is a string
    if callable(models.__dict__[args.model]):
        m =  models.__dict__[args.model](num_classes=len(args.classes))
        norm = Normalize(args.mean, args.std, inplace=True)
        return nn.Sequential(norm, m)
    raise ValueError("Unknown model arg: {}".format(args.model))
```

A little trick that saves me a lot of headache is putting the normalization transform as the first layer in the model. Now I just pass in image tensors normalized to [0, 1] and don't have to worry about what normalization the model was trained on because that information is stored in the checkpoint automatically by pytorch lightning.


# Training

The final output of the model goes through a softmax so cross entropy loss is used.

All the arguments:
```
--seed=42                   # int, deterministic seed, cudnn.deterministic is always set True by deafult
--root=data/full            # str, ImageFolder root, REQUIRED
--name=default              # str, Tensorboard name and log folder name
--workers=0                 # int, Dataloader num_workers, good practice is to use number of cpu cores or less
--ngpu=1                    # int, number of gpus to train on
--benchmark                 # store_true, set cudnn.benchmark
--precision=32              # int, 32 for full precision and 16 uses pytorch amp
--batch=16                  # int, batch size
--split=0.1                 # float, validation size as a percentage of the training set size
--imbalance                 # store_true, use my imbalance sampler
--mean=[]                   # 3 floats, dataset mean, default is imagenet [0.485, 0.456, 0.406]
--std=[]                    # 3 floats, dataset std, default is imagenet [0.229, 0.224, 0.225]
--model=resnet18            # str, torchvision model name
--input_size=224            # int, input square size in pixels
--epochs=20                 # int, number of epochs
--opt=adam                  # str, use sgd, adam, or adamw
--lr=4e-3                   # float, learning rate
--momentum=0.9              # float, sgd with momentum
--nesterov                  # store_true, sgd with nestrov acceleration
--weight_decay=0.0          # float, weight decay for sgd and adamw
--accumulate=1              # int, number of gradient accumulation steps, simulate larger batches when >1
--scheduler=None            # str, use step, plateau, or exp schedulers
--lr_gamma=0.2              # float, multiply the learning rate by gamma on the scheduler step
--milestones 10 15          # ints, step scheduler milestones
--plateau_patience=20       # int, plateau scheduler patience, monitoring the train loss
--save_top_k=1              # int, save topk model checkpoints
--save_monitor='val_loss'   # str, use val_loss or val_accuracy to find topk models
--early_stop=None           # str, use loss or acc for callback
--early_stop_patience=20    # int, patience for the early stop callback
--cutmix=0                  # float, cutmix beta value, 0 is no cutmix
--aug_scale=0.08            # float, lower bound for RandomResizedCrop
```

This is the tensorboard accruacy graph for 2 resnet18 runs. The dark red is random sampler and the light blue is a imbalance sampler which upsamples the minority classes to make uniform labels in each batch.
![resnet18_acc](/data/readme/resnet18_acc.png)

Here is the command to replicate these training runs:

Dark red line:
```
python train.py --root=data/full448 --seed=42 --workers=6 --split=0.2 --epochs=200 --batch=32 --model=resnet18 --opt=adamw --lr=4e-3 --weight_decay=5e-4 --scheduler=step --milestones 150 190 --lr_gamma=0.1
```

Light blue line, add --imbalance:
```
python train.py --root=data/full448 --seed=42 --workers=6 --split=0.2 --epochs=200 --batch=32 --model=resnet18 --opt=adamw --lr=4e-3 --weight_decay=5e-4 --scheduler=step --milestones 150 190 --lr_gamma=0.1 --imbalance
```

The imbalance sampling seemed to underperform, but that's complete speculation. But I don't have a statistical sample of runs, just these 2 runs here.


# Training Methodolgy

1. Adam converges faster than SGD. Shocking. I still want to test SGD with different learning rates, but for now just use adam with a learning rate of 4e-3.

2. Weight Decay helps. You have to use adamw. Try 5e-4 or 1e-4. I found a function on pytorch forms that removes weight decay from batchnorm and bias terms. Check configure_optimizers in the model.

3. Training image transforms. I used the torchvision transforms. I usually start with RandomResizedCrop and RandomHorizontalFlip. Then, I added ColorJitter, RandomVerticalFlip, and RandomGrayscale. The biggest changes came with what I would call "optical transforms"; these where RandomPerspective, RandomAffine, and RandomRotation. Increasing rotation and shear had the largest improvement in validation accuracy.


# Evaluation

## Everything is in here -> [evaluation.ipynb](evaluation.ipynb)

One code block prints statistics for each model like this.
```
    Model               Arch  Accuracy Precision    Recall  F1 Score
        0           resnet18     94.31     94.42     94.31     94.27
        1 mobilenet_v3_large     88.61     88.99     88.61     88.47
        2 mobilenet_v3_small     93.89     93.97     93.89     93.87
        3 shufflenet_v2_x0_5     94.17     94.26     94.17     94.13
        4 shufflenet_v2_x1_0     95.83     95.91     95.83     95.83
        5 shufflenet_v2_x1_5     95.97     96.07     95.97     95.97
        6 shufflenet_v2_x2_0     95.83     95.94     95.83     95.83
 Ensemble                        96.25     96.35     96.25     96.22
```
Notice how the ensemble score is higher than any of the individual models. The ensemble is the mean of the softmax from each model. I combine the outputs this way so it has a sum of 1.

One code block prints the ensemble predictions using sklearn.metrics.classification_report.
```
              precision    recall  f1-score   support

   cardboard     0.9833    0.9833    0.9833       120
       glass     0.9231    1.0000    0.9600       120
       metal     0.9669    0.9750    0.9710       120
       paper     0.9524    1.0000    0.9756       120
     plastic     0.9643    0.9000    0.9310       120
       trash     0.9910    0.9167    0.9524       120

    accuracy                         0.9625       720
   macro avg     0.9635    0.9625    0.9622       720
weighted avg     0.9635    0.9625    0.9622       720
```
Glass and paper had perfect recall; recall is the probability of detection (true predictions over the number of targets in the class).


The confusion matrix is done with sklearn.metrics.confusion_matrix and seaborn.heatmap.

![confusion_matrix](/data/readme/confusion_matrix.png)

This matrix tells us more than accuracy, precision, and recall. For instance, the largest of diagonal value is 8. This square tells us that the model gets confused about plastic and glass.