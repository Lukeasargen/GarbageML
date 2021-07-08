import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.transforms import Normalize
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, precision_recall, f1

from util import LabelSmoothing


def get_model(args):
    # args.model is a string
    if callable(models.__dict__[args.model]):
        m = models.__dict__[args.model](num_classes=args.num_classes)
        norm = Normalize(args.mean, args.std, inplace=True)
        return nn.Sequential(norm, m)
    raise ValueError("Unknown model arg: {}".format(args.model))


class GarbageModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super(GarbageModel, self).__init__()
        self.save_hyperparameters()
        self.hparams.num_classes = len(self.hparams.classes)
        self.model = get_model(self.hparams)
        self.scheduler = None
        assert 0 <= self.hparams.label_smoothing < (self.hparams.num_classes-1)/self.hparams.num_classes
        self.criterion = LabelSmoothing(self.hparams.label_smoothing)

    def forward(self, x):
        """ Inference Method Only"""
        return torch.softmax(self.model(x), dim=1)

    def batch_step(self, batch):
        """ Used in train and validation """
        data, target = batch
        if self.hparams.cutmix>0 and self.training:
            lam = np.random.beta(self.hparams.cutmix, self.hparams.cutmix)
            rand_index = torch.randperm(data.size()[0]).to(data.device)
            target_a = target
            target_b = target[rand_index]
            # Now the bboxes for the input and mask
            _, _, w, h = data.size()
            cut_rat = np.sqrt(1.0 - lam)
            cut_w, cut_h = int(w*cut_rat), int(h*cut_rat)  # Box size
            cx, cy = np.random.randint(w), np.random.randint(h)  # Box center
            bbx1 = np.clip(cx - cut_w // 2, 0, w)
            bbx2 = np.clip(cx + cut_w // 2, 0, w)
            bby1 = np.clip(cy - cut_h // 2, 0, h)
            bby2 = np.clip(cy + cut_h // 2, 0, h)
            data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
            # Adjust the classification loss based on pixel area ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w*h))
            logits = self.model(data)
            loss = self.criterion(logits, target_a)*lam + self.criterion(logits, target_b)*(1.0-lam)
        else:
            logits = self.model(data)
            loss = self.criterion(logits, target)
        # Metrics
        pred = torch.argmax(logits, dim=1)
        acc = accuracy(pred, target)
        avg_precision, avg_recall = precision_recall(pred, target, num_classes=self.hparams.num_classes,
                                                        average="macro", mdmc_average="global")
        weighted_f1 = f1(pred, target, num_classes=self.hparams.num_classes,
                            threshold=0.5, average="weighted")
        metrics = {
            "loss": loss,  # attached to computation graph, not necessary in validation, but I'm to lazy to fix
            "accuracy": acc,
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "weighted_f1": weighted_f1,
        }
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.batch_step(batch)
        for k, v in metrics.items():
            key = "{}/train".format(k)
            val = metrics[k].detach().item()
            self.logger.experiment.add_scalar(key, val, global_step=self.global_step)
        return metrics

    def training_epoch_end(self, outputs):
        # Add graph to tensorboard
        if self.current_epoch == 0:
            sample = torch.rand((1, 3, self.hparams.input_size, self.hparams.input_size), device=self.device)
            self.logger.experiment.add_graph(self.model, sample)
        
        # Parameter histograms
        # Too long to reload tensorboard, so commented out
        # for name, params in self.named_parameters():
        #     try:
        #         self.logger.experiment.add_histogram(name, params, self.current_epoch)
        #         self.logger.experiment.add_histogram(f'{name}.grad', params.grad, self.current_epoch)
        #     except Exception as e:
        #         pass

        # Calculating epoch metrics 
        for k in outputs[0].keys():
            key = "{}/train_epoch".format(k)
            val = torch.stack([x[k] for x in outputs]).mean().detach().item()
            self.logger.experiment.add_scalar(key, val, global_step=self.current_epoch)
        
    def validation_step(self, batch, batch_idx):
        metrics = self.batch_step(batch)
        return metrics

    def validation_epoch_end(self, outputs): 
        # Calculate epoch metrics 
        for k in outputs[0].keys():
            key = "{}/val_epoch".format(k)
            val = torch.stack([x[k] for x in outputs]).mean().detach().item()
            self.logger.experiment.add_scalar(key, val, global_step=self.current_epoch)
            # Use log for the checkpoint callback
            if k=="loss":
                avg_loss = val  # For plateau scheduler
                self.log('val_loss', val)
            if k=="accuracy":
                self.log('val_accuracy', val)

        # Log lr
        lr = self.optimizers().param_groups[0]['lr']
        self.logger.experiment.add_scalar('Learning Rate', lr, global_step=self.current_epoch)

        # Step scheduler
        if self.scheduler:
            if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                self.scheduler.step(avg_loss)
            elif type(self.scheduler) in [torch.optim.lr_scheduler.MultiStepLR, torch.optim.lr_scheduler.ExponentialLR]:
                self.scheduler.step()


    def configure_optimizers(self):

        """https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3"""
        def add_weight_decay(module, weight_decay):
            decay = []
            no_decay = []
            for name, param in module.named_parameters():
                if param.requires_grad:
                    if len(param.shape) == 1:  # Bias and bn parameters
                        no_decay.append(param)
                    else:
                        decay.append(param)
            return [{'params': no_decay, 'weight_decay': 0.0},
                    {'params': decay, 'weight_decay': weight_decay}]

        if self.hparams.weight_decay != 0:
            params = add_weight_decay(self.model, self.hparams.weight_decay)
        else:
            params = self.model.parameters()

        if self.hparams.opt == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.hparams.lr, momentum=self.hparams.momentum,
                            nesterov=self.hparams.nesterov)
        elif self.hparams.opt == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.hparams.lr)
        elif self.hparams.opt == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=self.hparams.lr)

        if self.hparams.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.milestones, gamma=self.hparams.lr_gamma)
        elif self.hparams.scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.hparams.lr_gamma, patience=self.hparams.patience, verbose=True)
        elif self.hparams.scheduler == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.lr_gamma)

        return optimizer
