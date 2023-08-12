# This file is a modified version of the original:
# https://github.com/mipypf/Vesuvius-Challenge/blob/winner_call/tattaka_ron/src/exp055/train.py

import argparse
import datetime
import math
import os
import random
import warnings
from functools import lru_cache, partial
from glob import glob
from typing import Callable, List, Tuple
from tqdm import tqdm
import albumentations as albu
import numpy as np
import PIL.Image as Image
import pytorch_lightning as pl
import timm
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule, callbacks
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks import TQDMProgressBar
from scipy.optimize import minimize
from timm.utils import ModelEmaV2
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

EXP_ID = "train_patch_based"
COMMENT = """2.5D + 3DCSN + no_mixup_epochs + channel shift
             + stochastic depth + mid_layer option + groups option
             + manifold mixup + cutmix + normal mixup
             + shapemarker + ema + label smoothing + fix seed + heavy aug
             + backbone lr option
             + other augmentation + grid_cutout + fbeta_loss
             """
BASE_DIR = "vesuvius_patches_32_5fold"

# import graphviz
# graphviz.set_jupyter_format('png')
# from torchview import draw_graph

class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        #bar = super().init_validation_tqdm()
        #bar.position=0
        #bar.leave=False
        #bar.dynamic_ncols = False
        ##bar.set_description("running validation...")
        #return bar
        return tqdm(
            desc=self.validation_description,
            position=0,
            leave=False,
        )


# typing module provides runtime support for type hints.
def get_transforms(train: bool = False) -> Callable:
    if train:
        return albu.Compose(
            [
                albu.Flip(p=0.5),
                albu.RandomRotate90(p=0.9),
                albu.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.2,
                    rotate_limit=15,
                    p=0.9,
                ),
                albu.OneOf(
                    [
                        albu.ElasticTransform(p=0.3),
                        albu.GaussianBlur(p=0.3),
                        albu.GaussNoise(p=0.3),
                        albu.OpticalDistortion(p=0.3),
                        albu.GridDistortion(p=0.1),
                        
                        # apply affine transformations that differ between local neighbourhoods.
                        albu.PiecewiseAffine(p=0.3),  # IAAPiecewiseAffine
                    ],
                    p=0.9,
                ),
                albu.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.3
                ),
                # convert image and mask to torch.Tensor. the numpy HWC image is converted to pytorch CHW tensor. If the image is in HW format (grayscale image), it will be converted to pytorch HW tensor.
                ToTensorV2(), 
            ]
        )

    else:
        return albu.Compose(
            [
                ToTensorV2(),
            ]
        )


def grid_cutout(volume, label, max_height: int = 2, max_width: int = 2, prob: float = 0.5):
    '''
    manual grid cut out i.e., assign zeros to random position in [:,H,W].
    '''
    # np.random.rand() => create a random value from a uniform distribution over [0, 1).
    if np.random.rand() < prob:
        # random.randint(low, high=None,...) => return random integers from low (inclusive) to high (exclusive).
        cut_h = np.random.randint(1, max_height + 1) * 32
        cut_w = np.random.randint(1, max_width + 1) * 32
        cut_y = np.random.randint(1, (volume.shape[1] - cut_h) // 32) * 32
        cut_x = np.random.randint(1, (volume.shape[2] - cut_w) // 32) * 32
        
        # volume.shape => torch.Size([65, 256, 256]), torch.Size([6, 256, 256])
        # cut_h, cut_w, cut_y, cut_x => 64, 32, 32, 128
        volume[:, cut_y : cut_y + cut_h, cut_x : cut_x + cut_w] = 0   # [:, 32 : 96, 128 : 160]
        label[:, cut_y : cut_y + cut_h, cut_x : cut_x + cut_w] = 0
        
    return volume, label


class PatchDataset(Dataset):
    def __init__(
        self,
        volume_paths: List[str],
        image_size: Tuple[int, int] = (256, 256),
        mode: str = "train",  # "train" | "valid" | "test"
        preprocess_in_model: bool = False,
        start_z: int = 8,
        end_z: int = -8,
        shift_z: int = 2,
    ):
        self.volume_paths = volume_paths
        self.image_size = image_size
        assert (image_size[0] % 32 == 0) and (image_size[1] % 32 == 0)
        self.mode = mode
        self.train = mode == "train"
        self.transforms = get_transforms(self.train)
        self.PATCH_SIZE = 32
        self.preprocess_in_model = preprocess_in_model
        self.start_z = start_z
        self.end_z = end_z
        self.shift_z = shift_z

    def __len__(self) -> int:
        if self.mode == "train":
            return 25000 # return len(self.volume_paths)# return 25000
        elif self.mode == "valid":
            return 24000 # return len(self.volume_paths)# return 24000
        else:
            return len(self.volume_paths)

    # .lru_cache(maxsize=128, typed=False) => wrap a function with a memoizing callable. Memoization is a technique to cache the results of expensive or I/O bound function calls and reuse them when the same function is called again with the same arguments. it can save time when a function is periodically called with the same arguments.
    # maxsize parameter, which determines the maximum number of most recent calls to be cached. 
    @lru_cache(maxsize=1024)
    def np_load(self, path: str) -> np.ndarray:
        return np.load(path)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        if self.train:
            np_load = np.load
            # np.random.choice(...) => generates a random sample from a given 1-D array.
            idx = np.random.choice(np.arange(len(self.volume_paths)))
        else:
            np_load = self.np_load
        
        volume = np.zeros((32, *self.image_size), dtype=np.float32)  # 65, 32
        
        label = np.zeros(self.image_size)
        volume_lt_path = self.volume_paths[idx]
        data_prefix = "/".join(volume_lt_path.split("/")[:-3])
        data_source = volume_lt_path.split("/")[-3]
        # data_prefix, data_source => vesuvius_patches_32_5fold/train, 3
        
        # volume_lt_path => vesuvius_patches_32_5fold/train/2/surface_volume/volume_13_134.npy
        y, x = volume_lt_path.split("/")[-1].split(".")[-2].split("_")[-2:]
        x = int(x)
        y = int(y)
        # y , x  => 13 (13th division/part of height), 134 (134th division/part of width)
        
        
        for i in range(self.image_size[0] // self.PATCH_SIZE):# 256//32 => 8
            for j in range(self.image_size[1] // self.PATCH_SIZE):
                
                volume_path = os.path.join(
                    data_prefix,
                    data_source,
                    f"surface_volume/volume_{y + i}_{x + j}.npy",
                )
                label_path = os.path.join(
                    data_prefix, data_source, f"label/label_{y + i}_{x + j}.npy"
                )
                if os.path.exists(volume_path):
                    # np_load(volume_path).shape => (65,32,32)
                    
                    volume[
                        :,
                        i * self.PATCH_SIZE : (i + 1) * self.PATCH_SIZE,
                        j * self.PATCH_SIZE : (j + 1) * self.PATCH_SIZE,
                    ] = np_load(volume_path)
                    
                    if os.path.exists(label_path):
                        label[
                            i * self.PATCH_SIZE : (i + 1) * self.PATCH_SIZE,
                            j * self.PATCH_SIZE : (j + 1) * self.PATCH_SIZE,
                        ] = np_load(label_path)
        
        # volume.shape => (65,256,256)
        if self.preprocess_in_model:
            # np.random.rand() => create a random value from a uniform distribution over [0, 1).
            if self.train and np.random.rand() < 0.5:
                # random.randint(low, high=None,...) => return random integers from low (inclusive) to high (exclusive).
                shift = np.random.randint(-self.shift_z, self.shift_z + 1)
            else:
                shift = 0
            # TODO: Add test for shift range
            volume = volume[
                self.start_z + shift : self.end_z + shift
            ]  # use mid 49th row/layer.    
            
            # volume.shape => (6,256,256)            
        
        volume = volume.transpose(1, 2, 0)        
        # volume.shape => (256,256,65) if self.preprocess_in_model==False otherwise (256,256,6)
        # label.shape => (256,256)
        
        aug = self.transforms(image=volume, mask=label)
        # aug["image"].shape, aug["mask"].shape => torch.Size([65, 256, 256]) torch.Size([256, 256])
        volume = aug["image"]
        label = aug["mask"][None, :]
        # label.shape => torch.Size([1, 256, 256])
        
        if self.train:
            volume, label = grid_cutout(
                volume=volume, label=label, max_height=2, max_width=2, prob=0.5 # 0.5 # 1 for my_ torchview
            )
        
        return (
            volume,
            label,
            x,
            y,
        )


class InkDetDataModule(LightningDataModule):
    def __init__(
        self,
        train_volume_paths: List[str],
        valid_volume_paths: List[str],
        image_size: int = 256,
        num_workers: int = 4,
        batch_size: int = 16,
        preprocess_in_model: bool = False,
        start_z: int = 8,
        end_z: int = -8,
        shift_z: int = 2,
    ):
        super().__init__()

        self._num_workers = num_workers
        self._batch_size = batch_size
        self.train_volume_paths = train_volume_paths
        self.valid_volume_paths = valid_volume_paths
        self.image_size = (image_size, image_size)
        self.preprocess_in_model = preprocess_in_model
        self.start_z = start_z
        self.end_z = end_z
        self.shift_z = shift_z
        
        # enable Lightning to store all the provided arguments under the self.hparams attribute. self.hparams is a convenient way to manage, access, and log hyperparameters throughout your training process, helping you keep track of how your model was configured and trained. These hyperparameters will also be stored within the model checkpoint, which simplifies model re-instantiation after training.
        self.save_hyperparameters(
            "num_workers",
            "batch_size",
            "image_size",
            "preprocess_in_model",
            "start_z",
            "end_z",
            "shift_z",
        )

    def create_dataset(self, mode: str = "train") -> PatchDataset:
        if mode == "train":
            return PatchDataset(
                volume_paths=self.train_volume_paths,
                image_size=self.image_size,
                mode=mode,
                preprocess_in_model=self.preprocess_in_model,
                start_z=self.start_z,
                end_z=self.end_z,
                shift_z=self.shift_z,
            )
        else:
            return PatchDataset(
                volume_paths=self.valid_volume_paths,
                image_size=self.image_size,
                mode=mode,
                preprocess_in_model=self.preprocess_in_model,
                start_z=self.start_z,
                end_z=self.end_z,
                shift_z=self.shift_z,
            )

    def __dataloader(self, mode: str = "train") -> DataLoader:
        """Train/validation loaders."""
        dataset = self.create_dataset(mode)
        return DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=(mode == "train"),
            drop_last=(mode == "train"),
            # worker_init_fn=lambda x: np.random.seed(np.random.get_state()[1][0] + x),
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self.__dataloader(mode="train")

    def val_dataloader(self) -> DataLoader:
        return self.__dataloader(mode="valid")

    def test_dataloader(self) -> DataLoader:
        return self.__dataloader(mode="test")

    # static method objects provide a way of defeating the transformation of function objects to method objects.
    # a static method does not receive an "implicit first argument" (self). we can even call it from an instance i.e., InkDetDataModule().add_model_specific_args(). 
    # a class method receives the class as "implicit first argument" (self), just like an instance method receives the instance.
    # the insertion of @staticmethod before a method definition, then, stops an instance from sending itself as an argument to the @staticmethod function.
    # decorator indicates that the method is a static method, meaning it doesn't require an instance of the class to be called. Static methods are defined within a class but don't have access to instance-specific attributes.    
    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser,) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("InkDetDataModule")
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            metavar="W",
            help="number of CPU workers",
            dest="num_workers",
        )
        parser.add_argument(
            "--batch_size",
            default=16,
            type=int,
            metavar="BS",
            help="number of sample in a batch",
            dest="batch_size",
        )
        parser.add_argument(
            "--image_size",
            default=256,
            type=int,
            metavar="IS",
            help="image size",
            dest="image_size",
        )
        return parent_parser


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).
    # Quality Focal Loss is an extension of the Focal Loss that incorporates additional factors to improve the handling of imbalanced classes in a binary classification problem.
    def __init__(
        self, loss_fcn: nn.Module, gamma: float = 1.5, alpha: float = 0.25, smooth=0.0
    ):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma # controls the focusing effect of the loss.
        self.alpha = alpha # that balances the loss for positive and negative examples. 
        self.reduction = loss_fcn.reduction
        self.smooth = smooth
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred: torch.Tensor, true: torch.Tensor):
        loss = self.loss_fcn(pred, true * (1 - (self.smooth / 0.5)) + self.smooth)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class FBetaLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, beta: float = 0.5):
        super().__init__()
        self.smooth = smooth
        self.beta = beta

    def forward(self, pred: torch.Tensor, true: torch.Tensor):
        pred_prob = torch.sigmoid(pred)
        if true.sum() == 0 and pred_prob.sum() == 0:
            return 0.0
        y_true_count = true.sum()
        ctp = (pred_prob * true).sum()
        cfp = (pred_prob * (1 - true)).sum()
        beta_squared = self.beta * self.beta

        c_precision = ctp / (ctp + cfp + self.smooth)
        c_recall = ctp / (y_true_count + self.smooth)
        fbeta = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall + self.smooth)
        )
        
        # convert fbeta score to fbeta loss.
        return 1 - fbeta


def downsample_conv(
    in_channels: int,
    out_channels: int,
    stride: int = 2,
):
    return nn.Sequential(
        *[
            nn.Conv3d(
                in_channels,
                out_channels,
                1,
                stride=(1, stride, stride),
                padding=0,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
        ]
    )


class ResidualConv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: int = 2,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, kernel_size=1, stride=1, bias=False),
            # torch.nn.Conv3d(in_channels, out_channels,.., groups=1) =>
            # groups= in_channels, each input channel is convolved with its own set of filters (of size (out_channels/ in_channels)).
            nn.Conv3d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=(1, stride, stride),
                padding=1,
                bias=False,
                groups=mid_channels,
            ),
        )
        self.bn2 = nn.BatchNorm3d(mid_channels)
        self.act2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.act3 = nn.ReLU(inplace=True)
        self.downsample = downsample_conv(
            in_channels,
            out_channels,
            stride=stride,
        )
        self.stride = stride
        self.zero_init_last()

    def zero_init_last(self):
        # initializes the weights of the last batch normalization layer (bn3) to zero. 
        if getattr(self.bn3, "weight", None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x: torch.Tensor):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


class InkDetModel(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet34",
        pretrained: bool = False,
        drop_rate: float = 0,
        drop_path_rate: float = 0,
        num_3d_layer: int = 3,
        in_chans: int = 7,
        preprocess_in_model: bool = False,
        start_z: int = 8, # 8 # my_
        end_z: int = -7, # 8 # -7 my_
        shift_z: int = 2,
        num_class: int = 1,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        
        # getattr(object, name, default) => return the value of the named attribute of object. If the named attribute does not exist, default is returned if provided.
        self.output_fmt = getattr(self.encoder, "output_fmt", "NHCW")
        
        self.in_chans = in_chans
        num_features = self.encoder.feature_info.channels()[-1]
        self.conv_proj = nn.Sequential(
            nn.Conv2d(num_features, 512, 1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.conv3d = nn.Sequential(
            *[
                ResidualConv3D(
                    512,
                    512,
                    512,
                    1,
                )
                for _ in range(num_3d_layer)
            ]
        )
        self.head = nn.Sequential(
            nn.Conv2d(
                512 * 2,
                512,
                1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(
                512,
                num_class,
                1,
            ),
        )
        self.preprocess_in_model = preprocess_in_model
        self.start_z = start_z
        self.end_z = end_z
        self.shift_z = shift_z

    def preprocess(self, img):
        
        if self.training and np.random.rand() < 0.5:
            shift = np.random.randint(-self.shift_z, self.shift_z + 1)
        else:
            shift = 0
        img = img[:, self.start_z + shift : self.end_z + shift]
        return img

    def forward_image_feats(self, img):
        
        if not self.preprocess_in_model:
            img = self.preprocess(img)
        mean = img.mean(dim=(1, 2, 3), keepdim=True)
        std = img.std(dim=(1, 2, 3), keepdim=True) + 1e-6
        img = (img - mean) / std
        
        
        # img.shape => torch.Size([16, 65, 256, 256]) # torch.Size([16, 6, 256, 256])
        bs, ch, h, w = img.shape
        # self.in_chans => 3
                
        #assert ch % self.in_chans == 0
        groups_3d = ch // self.in_chans
        
        img = img.reshape((bs, groups_3d, self.in_chans, h, w))
        

        if self.training:
            ch_arr = list(range(img.shape[2]))
            # ch_arr => [0, 1, 2]
            
            ch_arr = [
                # random.sample(population, k, ...) => Return a k length list of unique elements chosen from the population sequence.
                random.sample(ch_arr, len(ch_arr)) if np.random.rand() < 0.2 else ch_arr
                for _ in range(img.shape[0])
            ]
            # ch_arr => [[2, 0, 1], [0, 1, 2], [2, 1, 0], [0, 2, 1], ..., [0, 1, 2], [2, 1, 0]]
            # len(ch_arr) => 16
            # img.shape => torch.Size([16, 2, 3, 256, 256]) i.e., 2*3=6
            for i, ca in enumerate(ch_arr):
                # for dimension 0 i.e., for every batch element, interchange the indices for dimension 2 in img.
                img[i] = img[i, :, ca] # this technique do not work if img is numpy array.
                
            # img.shape => torch.Size([16, 2, 3, 256, 256])

        img = img.reshape(bs * groups_3d, self.in_chans, h, w)
        img_feat = self.encoder(img)[-1]
        if self.output_fmt == "NHWC":
            img_feat = img_feat.permute(0, 3, 1, 2).contiguous()
        img_feat = self.conv_proj(img_feat)  # (bs * groups_3d, 512, h, w)
        _, ch, h, w = img_feat.shape
        img_feat = img_feat.reshape(bs, groups_3d, ch, h, w).transpose(
            1, 2
        )  # (bs, ch, groups_3d, h, w)
        img_feat = self.conv3d(img_feat)  # (bs, ch, groups_3d, h, w)
        img_feat = torch.cat([img_feat.mean(2), img_feat.max(2)[0]], 1)
        return img_feat

    def forward(
        self,
        img: torch.Tensor,
    ):
        """
        img: (bs, ch, h, w)
        """
        img_feat = self.forward_image_feats(img)
        return self.head(img_feat)


class Mixup(object):
    def __init__(self, p: float = 0.5, alpha: float = 0.5):
        self.p = p
        self.alpha = alpha
        self.lam = 1.0
        self.do_mixup = False

    def init_lambda(self):
        if np.random.rand() < self.p:
            self.do_mixup = True
        else:
            self.do_mixup = False
        if self.do_mixup and self.alpha > 0.0:
            # random.beta(a, b, ...) => draw samples from a Beta distribution.
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1.0

    def reset_lambda(self):
        self.lam = 1.0


def fbeta_score(
    targets: np.ndarray, preds: np.ndarray, beta: float = 0.5, smooth: float = 1e-6
):
    if targets.sum() == 0 and preds.sum() == 0:
        return 1.0
    
    y_true_count = targets.sum()
    # ctp => correct true positives.
    ctp = (preds * targets).sum()
    cfp = (preds * (1 - targets)).sum()
    beta_squared = beta * beta

    # c_precision => conditional precision
    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    fbeta = (
        (1 + beta_squared)
        * (c_precision * c_recall)
        / (beta_squared * c_precision + c_recall + smooth)
    )

    return fbeta


def func_percentile(y_true: np.ndarray, y_pred: np.ndarray, t: List[float]):
    score = fbeta_score(
        y_true,
        # .clip(a, a_min, a_max, ...) => clip (limit) the values in an array. given an interval, values outside the interval are clipped to the interval edges.
        # .quantile(a, q, axis=None, ...) => compute the q-th quantile of the data along the specified axis.
        (y_pred > np.quantile(y_pred, np.clip(t[0], 0, 1))).astype(int),
        beta=0.5,
    )
    
    # - => concept of "negation" to convert minization to maximization. 
    return -score


def find_threshold_percentile(y_true: np.ndarray, y_pred: np.ndarray):
    # initial guess for the threshold value.
    x0 = [0.5] 
    
    # optimized threshold.
    # partial(...) => create a new function (from "func_percentile") that only requires two arguments (y_true and y_pred) instead of three (y_true, y_pred, and t). The t argument is being "fixed" or "frozen" using functools.partial. When this new function is called, it only expects the remaining arguments (t) that are not already provided. x0 is passed as t argument by the minimze func of scipy during call.
    # minimize(fun, x0, ...) => minimization of scalar function "fun" of one or more variables.
    threshold = minimize(partial(func_percentile, y_true, y_pred,),
        x0,
        method="nelder-mead",
    ).x[0]
    
    # .clip(a, a_min, a_max, ...) => clip (limit) the values in an array. given an interval, values outside the interval are clipped to the interval edges.
    return np.clip(threshold, 0, 1)


def rand_bbox(size, lam):
    '''
    random bounding box coordinates for performing a type of data augmentation known as "CutMix". width and height of the cut is based on the input lam value. CutMix augmentation, is typically used to generate random bounding box coordinates for overlaying a portion of one image onto another during training. 
    '''
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # random integers from the “discrete uniform” distribution.
    # cx,cy => random coordinates for the center of the cut.
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # .clip() => to ensure that the bounding box coordinates remain within the image dimensions.
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class InkDetLightningModel(pl.LightningModule):
    def __init__(
        self,
        valid_fragment_id: str,
        model_name: str = "resnet34",
        pretrained: bool = False,
        drop_rate: float = 0,
        drop_path_rate: float = 0,
        num_3d_layer: int = 3,
        in_chans: int = 7,
        preprocess_in_model: bool = False, # my_ True
        start_z: int = 8,
        end_z: int = -8,
        shift_z: int = 2,
        mixup_p: float = 0.0,
        mixup_alpha: float = 0.5,
        no_mixup_epochs: int = 0,
        lr: float = 1e-3,
        backbone_lr: float = None,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.backbone_lr = backbone_lr if backbone_lr is not None else lr
        self.__build_model(
            model_name=model_name,
            pretrained=pretrained,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            num_3d_layer=num_3d_layer,
            preprocess_in_model=preprocess_in_model,
            start_z=start_z,
            end_z=end_z,
            shift_z=shift_z,
            in_chans=in_chans,
        )
        self.mixupper = Mixup(p=mixup_p, alpha=mixup_alpha)
        self.mixup_alpha = mixup_alpha
        self.no_mixup_epochs = no_mixup_epochs        
        
        # torch.tensor(np.array(Image.open(f"vesuvius-challenge-ink-detection-5fold/train/{valid_fragment_id}/inklabels.png").convert("1")).astype(np.float32)).shape => torch.Size([8181, 6330])
        
        # torch.tensor(np.array(Image.open(f"vesuvius-challenge-ink-detection-5fold/train/{valid_fragment_id}/inklabels.png").convert("1")).astype(np.float32))[None, None].shape => 
        # torch.Size([1, 1, 8181, 6330])
        
        # interpolation or resampling of tensors. It allows you to adjust the spatial dimensions (width, height, depth) of a tensor to a desired size or scale factor. 
        self.y_valid = F.interpolate(
            torch.tensor(
                np.array(
                    Image.open(
                        f"vesuvius-challenge-ink-detection-5fold/train/{valid_fragment_id}/inklabels.png"
                    ).convert("1")
                ).astype(np.float32)
            )[None, None],
            scale_factor=1 / 32,
            mode="bilinear",
            align_corners=True,
        )[0, 0].numpy()
        
        # self.y_valid.shape => (255, 197)
        
        # .zeros_like(a, dtype=None, ...) => return an array of zeros with the same shape and "type" as a given array. "type" can be overided with dtype.        
        self.p_valid = np.zeros_like(self.y_valid, dtype=np.float32)
        self.count_pix = np.zeros_like(self.y_valid, dtype=np.float32)
        self.save_hyperparameters()

    def __build_model(
        self,
        model_name: str = "resnet34",
        pretrained: bool = False,
        drop_rate: float = 0,
        drop_path_rate: float = 0,
        num_3d_layer: int = 3,
        in_chans: int = 7,
        preprocess_in_model: bool = False,
        start_z: int = 8,
        end_z: int = -8,
        shift_z: int = 2,
    ):
        self.model = InkDetModel(
            model_name=model_name,
            pretrained=pretrained,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            num_3d_layer=num_3d_layer,
            in_chans=in_chans,
            preprocess_in_model=preprocess_in_model,
            start_z=start_z,
            end_z=end_z,
            shift_z=shift_z,
            num_class=1,
        )
        
        # When training a model, it is often beneficial to maintain moving averages of the trained parameters. Evaluations that use averaged parameters sometimes produce significantly better results than the final trained values.
        self.model_ema = ModelEmaV2(self.model, decay=0.99)
        self.criterions = {
            "bce": nn.BCEWithLogitsLoss(),
            "fbeta": FBetaLoss(),
        }

    def calc_loss(self, outputs: torch.Tensor, labels: torch.Tensor):
        losses = {}
        true = F.interpolate(
            labels["targets"],
            scale_factor=1 / 32,
            mode="bilinear",
            align_corners=True,
        )
        # true.shape => torch.Size([16, 1, 8, 8])
        
        smooth = 0.1
        true = true * (1 - (smooth / 0.5)) + smooth
        losses["bce"] = self.criterions["bce"](
            outputs["logits"],
            true,
        )
        
        # losses["bce"].shape => torch.Size([])
        # losses["bce"] => tensor(0.7165, device='cuda:0', dtype=torch.float64, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>) 
        
        losses["fbeta"] = self.criterions["fbeta"](outputs["logits"], true)
        losses["loss"] = losses["bce"] + losses["fbeta"]
        return losses

    def training_step(self, batch, batch_idx):
        self.model_ema.update(self.model)
        step_output = {}
        outputs = {}
        loss_target = {}
        self.mixupper.init_lambda()
        volume, label, _, _ = batch
        if (
            self.mixupper.do_mixup
            and self.current_epoch < self.trainer.max_epochs - self.no_mixup_epochs
        ):
            if np.random.rand() < 0.5:
                # manifold mixup. instead of interpolating between examples in the input space, "manifold mixup" interpolates in the feature space learned by the model.
                img_feat = self.model.forward_image_feats(volume)
                img_feat = self.mixupper.lam * img_feat + (1 - self.mixupper.lam) * img_feat.flip(0)
                outputs["logits"] = self.model.head(img_feat)
            else:
                # normal mixup.
                volume = self.mixupper.lam * volume + (1 - self.mixupper.lam) * volume.flip(0)
                outputs["logits"] = self.model(volume)
        else:
            if (
                np.random.rand() < 0.5
                and self.current_epoch < self.trainer.max_epochs - self.no_mixup_epochs
            ):
                # CutMix augmentation, is typically used to generate random bounding box coordinates for overlaying a portion of one image onto another during training.
                lam = (np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 0)
                bbx1, bby1, bbx2, bby2 = rand_bbox(volume.size(), lam)
                
                # .flip(dims) => reverse the order of an n-D tensor along given axis in dims.
                volume[:, :, bbx1:bbx2, bby1:bby2] = volume.flip(0)[
                    :, :, bbx1:bbx2, bby1:bby2
                ]
                label[:, :, bbx1:bbx2, bby1:bby2] = label.flip(0)[
                    :, :, bbx1:bbx2, bby1:bby2
                ]
                
            outputs["logits"] = self.model(volume)

        loss_target["targets"] = label
        losses = self.calc_loss(outputs, loss_target)

        if (
            self.mixupper.do_mixup
            and self.current_epoch < self.trainer.max_epochs - self.no_mixup_epochs
        ):
            loss_target["targets"] = loss_target["targets"].flip(0)
            losses_b = self.calc_loss(outputs, loss_target)
            for key in losses:
                losses[key] = (
                    self.mixupper.lam * losses[key]
                    + (1 - self.mixupper.lam) * losses_b[key]
                )
                
        step_output.update(losses)
        self.log_dict(
            dict(
                train_loss=losses["loss"],
                train_bce_loss=losses["bce"],
                train_fbeta_loss=losses["fbeta"],
            )
        )
        return step_output

    def validation_step(self, batch, batch_idx):
        
        step_output = {}
        outputs = {}
        loss_target = {}                
        volume, label, x, y = batch # x (xth division/part of width), y (yth division/part of height)
        # x.shape, y.shape => torch.Size([16]), torch.Size([16])
        
        # volume.shape => torch.Size([16, 6, 256, 256])
        outputs["logits"] = self.model_ema.module(volume)

        loss_target["targets"] = label
        
        # outputs["logits"].shape, loss_target["targets"].shape => torch.Size([16, 1, 8, 8]), torch.Size([16, 1, 256, 256])
        
        losses = self.calc_loss(outputs, loss_target)

        step_output.update(losses)
        pred_batch = (torch.sigmoid(outputs["logits"]).detach().to(torch.float32).cpu().numpy())
        
        # pred_batch.shape  => (16, 1, 8, 8)
        for xi, yi, pi in zip(x, y, pred_batch,):
            # pi.shape => (1, 8, 8)
            # self.y_valid[yi : yi + pred_batch.shape[-2], xi : xi + pred_batch.shape[-1],].shape => (8, 8)
            
            # yi : yi + pred_batch.shape[-2] => from the "yi"th division/part next 8 patches are combined to form height 256.
            # xi : xi + pred_batch.shape[-1] => from the "xi"th division/part next 8 patches are combined to form width 256.
            y_lim, x_lim = self.y_valid[yi : yi + pred_batch.shape[-2], xi : xi + pred_batch.shape[-1],].shape
            
            # y_lim, x_lim => 8, 8
            
            self.p_valid[yi : yi + pred_batch.shape[-2], xi : xi + pred_batch.shape[-1]] += pi[0, :y_lim, :x_lim]
            
            self.count_pix[yi : yi + pred_batch.shape[-2], xi : xi + pred_batch.shape[-1]] += np.ones_like(pi[0, :y_lim, :x_lim])
        
        self.log_dict(
            dict(
                val_loss=losses["loss"],
                val_bce_loss=losses["bce"],
                val_fbeta_loss=losses["fbeta"],
            )
        )
        return step_output

    def on_validation_epoch_end(self):
        self.p_valid /= self.count_pix
        
        # .nan_to_num(x, ...) => replace NaN with zero and infinity with large finite numbers (default behaviour) or with the numbers defined by the user using the nan, posinf and/or neginf keywords.
        self.p_valid = np.nan_to_num(self.p_valid)
        
        # self.count_pix => [[0. 0. 0. ... 0. 0. 0.][0. 0. 0. ... 0. 0. 0.] .... [0. 0. 0. ... 0. 0. 0.]]                
        self.count_pix = self.count_pix > 0
        # self.count_pix => [False False ... False False][False False ... False False] ... [False False ... False False]]
        
        
        # self.count_pix.shape => (255, 197)
        # self.count_pix.reshape(-1).shape => (50235,)
        # np.where(self.count_pix.reshape(-1))[0].shape => (808,)
        
        # .where(condition, [x, y, ]/) => return elements chosen from x or y depending on condition. When only condition is provided, returns indices of True values.
        # choose p_valid values from indices where there is a count.
        p_valid = self.p_valid.reshape(-1)[np.where(self.count_pix.reshape(-1))]
        y_valid = self.y_valid.reshape(-1)[np.where(self.count_pix.reshape(-1))]
        
        threshold = find_threshold_percentile(y_valid, p_valid)
        p_valid = p_valid > np.quantile(p_valid, threshold)

        score = fbeta_score(y_valid, p_valid, beta=0.5)
        self.p_valid = np.zeros_like(self.y_valid, dtype=np.float32)
        self.count_pix = np.zeros_like(self.y_valid, dtype=np.float32)
        self.log_dict(
            dict(
                val_fbeta_score=score,
                val_threshold=threshold,
            ),
            sync_dist=True,
        )
        
        print(f"\nepoch {self.current_epoch}: fbeta_score {score}, threshold {threshold}")

    def get_optimizer_parameters(self):
        # parameter names that should not have weight decay applied. 
        no_decay = ["bias", "gamma", "beta"]        
        
        # parameters of specific layers (of model) are organized in respective dictionaries.
        optimizer_parameters = [
            # used for parameters from the self.model.encoder.
            {
                # n => stem_0.weight, stem_0.bias, stem_1.weight, stem_1.bias, stages_0.blocks.0.gamma, stages_0.blocks.0.conv_dw.weight ...
                # p => tensor containing parameter values.
                "params": [
                    p
                    for n, p in self.model.encoder.named_parameters()
                    if any(nd in n for nd in no_decay) # any() => return True if any element of the iterable is true.
                ],
                "weight_decay": 0,
                "lr": self.backbone_lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model.encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0001,
                "lr": self.backbone_lr,
            },
            # used for parameters from self.model.conv_proj, self.model.conv3d, and self.model.head.
            {
                "params": [
                    p
                    for n, p in list(self.model.conv_proj.named_parameters())
                    + list(self.model.conv3d.named_parameters())
                    + list(self.model.head.named_parameters())
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
                "lr": self.lr,
            },
            {
                "params": [
                    p
                    for n, p in list(self.model.conv_proj.named_parameters())
                    + list(self.model.conv3d.named_parameters())
                    + list(self.model.head.named_parameters())
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0001,
                "lr": self.lr,
            },
        ]
        
        # Different layers of a neural network might require different learning rates based on factors such as their depth, complexity, and sensitivity to updates. For example, you might want to use a smaller learning rate for earlier layers of a deep neural network to prevent large updates that could disrupt their pretrained features, while using a larger learning rate for later layers to allow faster convergence.
        
        return optimizer_parameters

    def configure_optimizers(self):
        self.warmup = True
        # self.get_optimizer_parameters() => dicts defining parameter groups to optimize.
        optimizer = AdamW(self.get_optimizer_parameters())
        max_train_steps = self.trainer.estimated_stepping_batches
        warmup_steps = math.ceil((max_train_steps * 2) / 100) if self.warmup else 0
        rank_zero_info(
            f"max_train_steps: {max_train_steps}, warmup_steps: {warmup_steps}"
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("InkDetLightningModel")
        parser.add_argument(
            "--model_name",
            default="resnet34",
            type=str,
            metavar="MN",
            help="Name (as in ``timm``) of the feature extractor",
            dest="model_name",
        )
        parser.add_argument(
            "--drop_rate", default=0.0, type=float, metavar="DR", dest="drop_rate"
        )
        parser.add_argument(
            "--drop_path_rate",
            default=0.0,
            type=float,
            metavar="DPR",
            dest="drop_path_rate",
        )
        parser.add_argument(
            "--num_3d_layer",
            default=3,
            type=int,
            metavar="N3L",
            dest="num_3d_layer",
        )
        parser.add_argument(
            "--in_chans",
            default=7,
            type=int,
            metavar="ICH",
            dest="in_chans",
        )
        parser.add_argument(
            "--mixup_p", default=0.0, type=float, metavar="MP", dest="mixup_p"
        )
        parser.add_argument(
            "--mixup_alpha", default=0.0, type=float, metavar="MA", dest="mixup_alpha"
        )
        parser.add_argument(
            "--no_mixup_epochs",
            default=0,
            type=int,
            metavar="NME",
            dest="no_mixup_epochs",
        )
        parser.add_argument(
            "--lr",
            default=1e-3,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="lr",
        )

        parser.add_argument(
            "--backbone_lr",
            default=None,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="backbone_lr",
        )

        return parent_parser


def get_args() -> argparse.Namespace:
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--seed",
        default=2022,
        type=int,
        metavar="SE",
        help="seed number",
        dest="seed",
    )
    parent_parser.add_argument(
        "--debug",
        action="store_true",
        help="1 batch run for debug",
        dest="debug",
    )
    dt_now = datetime.datetime.now()
    parent_parser.add_argument(
        "--logdir",
        default=f"{dt_now.strftime('%Y%m%d-%H-%M-%S')}",
    )
    parent_parser.add_argument(
        "--fold",
        type=int,
        default=0,
    )
    parent_parser.add_argument(
        "--gpus", type=int, default=4, help="number of gpus to use"
    )
    parent_parser.add_argument(
        "--epochs", default=10, type=int, metavar="N", help="total number of epochs"
    )
    parent_parser.add_argument(
        "--preprocess_in_model",
        default=True,
        action="store_true",
        help="preprocess_in_model",
        dest="preprocess_in_model",
    )
    parent_parser.add_argument(
        "--start_z",
        default=8,
        type=int,
        metavar="SZ",
        help="start_z layer",
        dest="start_z",
    )
    parent_parser.add_argument(
        "--end_z",
        default=-8,
        type=int,
        metavar="EZ",
        help="end_z layer",
        dest="end_z",
    )
    parent_parser.add_argument(
        "--shift_z",
        default=2,
        type=int,
        metavar="SHZ",
        help="shift_z layer",
        dest="shift_z",
    )
    parser = InkDetLightningModel.add_model_specific_args(parent_parser)
    parser = InkDetDataModule.add_model_specific_args(parser)
    return parser.parse_args()


def main(args):
    pl.seed_everything(args.seed, workers=True)
    if not args.debug:
        warnings.simplefilter("ignore")
    fragment_ids = ["1", "2", "3", "4", "5"]
    for i, valid_idx in enumerate(fragment_ids):
        if args.fold != i: continue
        
        train_volume_paths = np.concatenate(
            [
                np.asarray(
                    sorted(
                        glob(
                            f"vesuvius_patches_32_5fold/train/{fragment_id}/surface_volume/**/*.npy",
                            recursive=True, # if recursive is true, the pattern “**” will match any files and zero or more directories, subdirectories and symbolic links to directories.
                        )
                    )
                )
                for fragment_id in fragment_ids
                if fragment_id != valid_idx
            ]
        )    # [:80] my_ 
        
        # len(train_volume_paths) => 121708
        
        valid_volume_paths = np.concatenate(
            [
                np.asarray(
                    sorted(
                        glob(
                            f"vesuvius_patches_32_5fold/train/{fragment_id}/surface_volume/**/*.npy",
                            recursive=True,
                        )
                    )
                )
                for fragment_id in fragment_ids
                if fragment_id == valid_idx
            ]
        ) # [:80] my_ 

        datamodule = InkDetDataModule(
            train_volume_paths=train_volume_paths,
            valid_volume_paths=valid_volume_paths,
            image_size=args.image_size,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            preprocess_in_model=args.preprocess_in_model,
            start_z=args.start_z,
            end_z=args.end_z,
            shift_z=args.shift_z,
        )
        model = InkDetLightningModel(
            valid_fragment_id=valid_idx,
            model_name=args.model_name,
            pretrained=True,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path_rate,
            num_3d_layer=args.num_3d_layer,
            in_chans=args.in_chans,
            preprocess_in_model=args.preprocess_in_model,
            start_z=args.start_z,
            end_z=args.end_z,
            shift_z=args.shift_z,
            mixup_p=args.mixup_p,
            mixup_alpha=args.mixup_alpha,
            no_mixup_epochs=args.no_mixup_epochs,
            lr=args.lr,
            backbone_lr=args.backbone_lr,
        )

        logdir = f"logs/exp_{EXP_ID}/{args.logdir}/fold{i}"
        print(f"logdir = {logdir}")
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename="best_loss",
            monitor="val_loss",
            save_top_k=1,
            save_last=True,
            save_weights_only=True,
            mode="min",
        )
        fbeta_checkpoint = callbacks.ModelCheckpoint(
            filename="best_fbeta",
            monitor="val_fbeta_score",
            save_top_k=3,
            save_last=False,
            save_weights_only=True,
            mode="max",
        )
        os.makedirs(os.path.join(logdir, "wandb"), exist_ok=True)
        
        if not args.debug:
#             wandb_logger = WandbLogger(
#                 name=f"exp_{EXP_ID}/{args.logdir}_fold{i}",
#                 save_dir=logdir,
#                 project="vesuvius-challenge-ink-detection",
#             )
            wandb_logger = CSVLogger(name=f"exp_{EXP_ID}/{args.logdir}_fold{i}", save_dir=logdir,)

        trainer = pl.Trainer(
            default_root_dir=logdir,
            sync_batchnorm=True,
            gradient_clip_val=2.0,
            precision=16,
            devices=args.gpus,
            accelerator="gpu",
            # strategy="ddp_find_unused_parameters_false",
            strategy="ddp",
            max_epochs=args.epochs,
            logger=wandb_logger if not args.debug else True,
            callbacks=[
                loss_checkpoint,
                fbeta_checkpoint,
                lr_monitor,
                LitProgressBar()
            ],
            fast_dev_run=args.debug,
            num_sanity_val_steps=0,
        )        

        
#         for d in datamodule.train_dataloader():
#             # d[0].shape, d[1].shape, d[2].shape, d[3].shape => torch.Size([16, 6, 256, 256]) torch.Size([16, 1, 256, 256]) torch.Size([16]) torch.Size([16])
#             draw_graph(model.model, input_data = [d[0]], expand_nested=True, save_graph=True).visual_graph            
#             break
#         return 1
        

        trainer.fit(model, datamodule=datamodule) 


if __name__ == "__main__":
    main(get_args())
