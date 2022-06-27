from collections import OrderedDict

from einops import rearrange
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.transforms import Normalize
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, precision_recall, f1

from util import LabelSmoothing


class CNN(torch.nn.Module):
    def __init__(self, features, fc, dropout=0.0):
        super(CNN, self).__init__()
        self.features = features
        self.fc = fc
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.flatten(x)
        return x

class ATTN(torch.nn.Module):
    def __init__(self, features, attention):
        super(ATTN, self).__init__()
        self.features = features
        self.attention = attention
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.flatten(x)
        return x

class ResidualAttentionBlock(torch.nn.Module):
    def __init__(self, embd, dim, heads, ff_multi=0, dropout=0):
        super(ResidualAttentionBlock, self).__init__()
        self.heads = heads
        self.ln = nn.LayerNorm(embd)
        self.to_qkv = nn.Linear(embd, 3*dim*heads, bias=False)
        self.attn_out = nn.Linear(dim*heads, embd, bias=False)
        self.scale_factor = dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.mlp = None
        if ff_multi>0:
            self.mlp = nn.Sequential(OrderedDict([
                ("in_proj", nn.Linear(embd, embd*ff_multi)),
                ("gelu", nn.GELU()),
                ("dropout", nn.Dropout(dropout)),
                ("out_proj", nn.Linear(embd*ff_multi, embd)),
                ("dropout", nn.Dropout(dropout)),
            ]))

    def attention(self, x):
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L186
        qkv = self.to_qkv(x)  # B T 3*D*H
        qkv = rearrange(qkv, 'b t (k d h) -> k b h t d', k=3, h=self.heads)  # 3 B H T D
        q, k, v = qkv.unbind(0)  # B H T D
        dots = torch.einsum('... i d , ... j d -> ... i j', q, k) * self.scale_factor  # B H T T
        attn = torch.softmax(dots, dim=-1)  # B H T T
        out = torch.einsum('... i j , ... j d -> ... i d', attn, v)  # B H T T
        out = rearrange(out, 'b h t d -> b t (h d)')  # B T D*H
        return self.attn_out(out)  # B T E

    def forward(self, x):
        # Doing the attention and mlp in parallel
        norm = self.ln(x)
        x = x + self.attention(norm)
        if self.mlp is not None:
            x = x + self.mlp(norm)
        x = self.dropout(x)
        return x

class AttentionClassifier(torch.nn.Module):
    def __init__(self, embd, dim, heads, layers, num_classes, ff_multi=0, dropout=0, attn_pos_size=2):
        super(AttentionClassifier, self).__init__()
        self.blocks = nn.Sequential(*[ResidualAttentionBlock(embd, dim, heads, ff_multi, dropout) for _ in range(layers)])
        self.out = nn.Linear(embd, num_classes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embd))  # B 1 E
        self.pos_embd = nn.Parameter(torch.randn(1, embd, attn_pos_size, attn_pos_size))  # B E H W, this is the corners of the position embedding
        self.dropout = nn.Dropout(dropout)
        # https://github.com/openai/CLIP/blob/main/clip/model.py#L300
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embd, std=0.01)
        proj_std, attn_std, fc_std = (embd ** -0.5) * ((2 * layers) ** -0.5), embd ** -0.5, (2 * embd) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.to_qkv.weight, std=attn_std)
            nn.init.normal_(block.attn_out.weight, std=proj_std)
            if block.mlp is not None:
                nn.init.normal_(block.mlp.in_proj.weight, std=fc_std)
                nn.init.normal_(block.mlp.out_proj.weight, std=fc_std)

    def forward(self, x):
        n, c, h, w = x.shape
        pos_embd = F.interpolate(self.pos_embd, size=(h,w), mode='bilinear', align_corners=True)
        x = x + pos_embd
        # TODO: mask out tokens
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = torch.cat([self.cls_token.expand(n, -1, -1), x], dim=1)  # B (HW+1) C
        x = self.dropout(x)
        x = self.blocks(x)
        out = self.out(x[:,0])
        return out

def get_model(args):
    norm = Normalize(args.mean, args.std, inplace=True)
    use_att = False
    model_name = args.model
    i = model_name.find("attn")  # Check for attention head
    if i>0:
        use_att = True
        model_name = model_name[:i-1]
    if callable(models.__dict__[model_name]):
        m = models.__dict__[model_name](pretrained=args.pretrained)
        # Get model features after pooling
        if "resnet" in model_name or "resnext" in model_name:
            layers = list(m.children())[:-2]  # Remove pooling and fc
        elif "shufflenet" in model_name:
            layers = list(m.children())[:-1]  # Remove fc
        elif "squeezenet" in model_name:
            layers = list(m.children())[:-1]  # Remove classifer
        elif "densenet" in model_name:
            layers = list(m.children())[:-1]  # Remove classifer
        elif "mobilenet_v2" in model_name:
            layers = list(m.children())[:-1]  # Remove classifer
        elif "mobilenet_v3" in model_name:
            layers = list(m.children())[:-2]  # Remove pooling and classifer
        elif "mnasnet" in model_name:
            layers = list(m.children())[:-1]  # Remove pooling and classifer
        else:
            raise ValueError("Model with pretrained not supported : {}".format(model_name))
        # Create a fake iamge to get the output dimension
        fake_img = torch.zeros(1, 3, args.input_size, args.input_size)
        yhat = nn.Sequential(*layers)(fake_img)
        _, final_dim, _, _ = yhat.shape
        features = nn.Sequential(norm, *layers)
        if use_att:
            if args.attn_embd!=final_dim:
                features.add_module("proj", nn.Conv2d(final_dim, args.attn_embd, kernel_size=1, bias=False))
            attention = AttentionClassifier(args.attn_embd, args.attn_dim, args.attn_heads,
                            args.attn_layers, args.num_classes, args.attn_ff_multi,
                            args.dropout, args.attn_pos_size)
            return ATTN(features, attention)
        else:
            fc = nn.Conv2d(final_dim, args.num_classes, kernel_size=1, bias=True)
            return CNN(features, fc, args.dropout)
    raise ValueError("Unknown model arg: {}".format(args.model))


class GarbageModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super(GarbageModel, self).__init__()
        self.save_hyperparameters()
        self.hparams.num_classes = len(self.hparams.classes)
        self.model = get_model(self.hparams)
        self.scheduler = None  # Set in configure_optimizers()
        self.opt_init_lr = None  # Set in configure_optimizers()
        assert 0 <= self.hparams.label_smoothing < (self.hparams.num_classes-1)/self.hparams.num_classes
        self.criterion = LabelSmoothing(self.hparams.label_smoothing)

    def forward(self, x):
        """ Inference Method Only"""
        return torch.softmax(self.model(x), dim=1)

    def batch_step(self, batch):
        """ Used in train and validation """
        data, target = batch
        if self.training and self.hparams.cutmix>0 and torch.rand(1) < self.hparams.cutmix_prob:
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

        pred = torch.argmax(logits, dim=1)
        acc = accuracy(pred, target)
        avg_precision, avg_recall = precision_recall(pred, target, num_classes=self.hparams.num_classes,
                                                        average="macro", mdmc_average="global")
        weighted_f1 = f1(pred, target, num_classes=self.hparams.num_classes,
                            threshold=0.5, average="weighted")
        metrics = {
            "loss": loss,  # attached to computation graph, not necessary in validation, but I'm to lazy to fix
            "accuracy": acc,
            "error": 1-acc,
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "weighted_f1": weighted_f1,
            "inv_f1": 1-weighted_f1,
        }
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.batch_step(batch)
        for k, v in metrics.items():
            key = "{}/train".format(k)
            self.log(key, v, on_step=True, on_epoch=True)

        if self.global_step == self.hparams.finetune_after and self.hparams.finetune_after>=0:
            for param in self.model.parameters():
                param.requires_grad = True

        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            opt = self.optimizers()
            lr_scale = min(1, float(self.trainer.global_step+1)/self.hparams.lr_warmup_steps)
            for pg, init_lr in zip(opt.param_groups, self.opt_init_lr):
                pg['lr'] = lr_scale*init_lr
        elif self.scheduler:
            if type(self.scheduler) in [torch.optim.lr_scheduler.MultiStepLR, torch.optim.lr_scheduler.ExponentialLR]:
                self.scheduler.step()

        lr = self.optimizers().param_groups[0]['lr']
        self.logger.experiment.add_scalar('Learning Rate/step', lr, global_step=self.global_step)

        return metrics["loss"]

    # def training_epoch_end(self, outputs):
        # avg_loss = torch.stack([x["loss"] for x in outputs]).mean().item()

        # Add graph to tensorboard
        # if self.current_epoch == 0:
        #     sample = torch.rand((1, 3, self.hparams.input_size, self.hparams.input_size), device=self.device)
        #     self.logger.experiment.add_graph(self.model, sample)
        
        # Parameter histograms
        # Too long to reload tensorboard, so commented out
        # for name, params in self.named_parameters():
        #     try:
        #         self.logger.experiment.add_histogram(name, params, self.current_epoch)
        #         self.logger.experiment.add_histogram(f'{name}.grad', params.grad, self.current_epoch)
        #     except Exception as e:
        #         pass
        
    def validation_step(self, batch, batch_idx):
        metrics = self.batch_step(batch)
        for k, v in metrics.items():
            key = "{}/val_epoch".format(k)
            self.log(key, v, on_step=False, on_epoch=True)
        return metrics["loss"]

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean().item()
        lr = self.optimizers().param_groups[0]['lr']
        self.logger.experiment.add_scalar('Learning Rate', lr, global_step=self.current_epoch)
        # Step scheduler
        if self.scheduler:
            if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                self.scheduler.step(avg_loss)


    def configure_optimizers(self):

        """https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3"""
        def add_weight_decay(module, weight_decay, lr):
            decay = []
            no_decay = []
            for name, param in module.named_parameters():
                if param.requires_grad:
                    if len(param.shape) == 1:  # Bias and bn parameters
                        no_decay.append(param)
                    else:
                        decay.append(param)
            return [{'params': no_decay, 'lr': lr,  'weight_decay': 0.0},
                    {'params': decay, 'lr': lr, 'weight_decay': weight_decay}]

        if self.hparams.pretrained:
            head_layer = self.model.fc if type(self.model)==CNN else self.model.attention
            if self.hparams.weight_decay != 0:
                params = add_weight_decay(head_layer, self.hparams.weight_decay, self.hparams.lr)
                # Don't weight decay on pretrained weights
                params += add_weight_decay(self.model.features, self.hparams.weight_decay, self.hparams.finetune_lr)
            else:
                params = [{'params': head_layer.parameters(), 'lr': self.hparams.lr},
                    {'params': self.model.features.parameters(), 'lr': self.hparams.finetune_lr}]
            # Pretrained weights are frozen until finetune_after
            for param in self.model.features.parameters():
                param.requires_grad = False
        else:
            # Not pretrained so all weights use the same hyperparameters
            if self.hparams.weight_decay != 0:
                params = add_weight_decay(self.model, self.hparams.weight_decay, self.hparams.lr)
            else:
                params = self.model.parameters()

        if self.hparams.opt == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.hparams.lr, momentum=self.hparams.momentum,
                            nesterov=self.hparams.nesterov)
        elif self.hparams.opt == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.hparams.lr)
        elif self.hparams.opt == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=self.hparams.lr)

        # Keep a copy of the initial lr for each group because this will get overwritten during warmup steps
        self.opt_init_lr = [pg['lr'] for pg in optimizer.param_groups]

        if self.hparams.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.milestones, gamma=self.hparams.lr_gamma)
        elif self.hparams.scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.hparams.lr_gamma, patience=self.hparams.plateau_patience, verbose=False)
        elif self.hparams.scheduler == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.lr_gamma)

        return optimizer
