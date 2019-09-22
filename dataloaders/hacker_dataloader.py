#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   hacker_dataloader.py.py    
@Contact :   alfredo2019love@outlook.com
@License :   (C)Copyright 2019-2020, DRL_Lab-Cheng-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/9/22 上午3:22   Cheng      1.0         init
'''
import random

import torch

import numpy as np
import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader
from PIL import Image
import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


def pil_loader(path, rgb=True):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if rgb:
            img = img.convert('RGB')
            img = np.asarray(img, dtype=np.float32)
            return img
        else:
            img = img.convert('L')
            img = np.asarray(img, dtype=np.float32)
            return img


iheight, iwidth = 720, 1280
# iheight, iwidth = 360, 640
alpha, beta = 0.02, 12.0  # NYU Depth, min depth is 0.02m, max depth is 10.0m
K = 80  # NYU is 68, but in paper, 80 is good


class HackerDataloader(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb'):
        super(HackerDataloader, self).__init__(root, type, sparsifier, modality)
        self.output_size = (256, 480)  # TODO
        data = type + '.txt'
        # data = 'test.txt' or 'val.txt'
        with open(os.path.join(root, data), 'r') as f:
            lines = f.readlines()
        self.root = root
        self.data = lines
        self.type = type
        self.loader = pil_loader
        self.size = (480, 256)  # (1280, 720)
        self.output_size = (480, 256)

    def __getitem__(self, index):
        sample = self.data[index].strip('\n').split()
        # im_path, gt_path
        im_path = os.path.join(self.root, sample[0])
        gt_path = os.path.join(self.root, sample[1])
        # print(im_path, gt_path)
        im = self.loader(im_path)
        gt = self.loader(gt_path, rgb=False)
        print('gt_path', gt_path)

        if self.type == 'train':
            input_np, depth_np = self.train_transform(im, gt)
        elif self.type == 'val':
            input_np, depth_np = self.val_transform(im, gt)
        elif self.type == 'test':
            input_np, depth_np = self.val_transform(im, gt)
        else:
            print("Error whih input type")
            exit(1)

        return input_np, depth_np

        # input_tensor = to_tensor(input_np)
        # """Convert a ``numpy.ndarray`` to tensor.
        # Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
        #
        # while input_tensor.dim() < 3:
        #     input_tensor = input_tensor.unsqueeze(0)
        # depth_tensor = to_tensor(depth_np)
        # depth_tensor = depth_tensor.unsqueeze(0)
        # return input_tensor, depth_tensor

        # print('input_tensor.size', input_tensor.size(), 'depth_tensor.size', depth_tensor.size())

        # print('depth_tensor---------------------------------------------', np.shape(depth_tensor))
        # rgb_np = np.asarray(im, dtype=np.float32)
        # depth_np = np.asarray(gt, dtype=np.float32)

        # if self.modality == 'rgb':
        #     input_np = rgb_np
        # elif self.modality == 'rgbd':
        #     input_np = self.create_rgbd(rgb_np, depth_np)
        # elif self.modality == 'd':
        #     input_np = self.create_sparse_depth(rgb_np, depth_np)
        # else:
        #     print('input_np is use before define')
        #     exit(-1)
        # print('input_np.size', np.shape(input_np))

    def train_transform(self, im, gt):
        # im = np.array(im).astype(np.float32)
        # gt = np.array(gt).astype(np.float32)

        # s = np.random.uniform(1.0, 1.5)  # random scaling
        angle = np.random.uniform(-17.0, 17.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
        # Upper_cor = random.randint(1, 464)
        # Left_cor = random.randint(1, 800)
        Upper_cor = random.randint(1, 360)
        Left_cor = random.randint(1, 640)
        # color_jitter = my_transforms.ColorJitter(0.4, 0.4, 0.4)

        transform = my_transforms.Compose([
            # my_transforms.Crop(Upper_cor, Left_cor, 256, 480),
            my_transforms.Crop(Upper_cor, Left_cor, 360, 640),
            # my_transforms.Resize(460 / 240, interpolation='bilinear'),
            my_transforms.Rotate(angle),
            # my_transforms.Resize(s),
            # my_transforms.CenterCrop(self.size),
            my_transforms.HorizontalFlip(do_flip)
        ])

        im_ = transform(im)
        # im_ = color_jitter(im_)

        gt_ = transform(gt)

        im_ = np.array(im_).astype(np.float32)
        gt_ = np.array(gt_).astype(np.float32)

        im_ /= 255.0
        gt_ /= 1000.0  # mm -> m
        im_ = to_tensor(im_)
        gt_ = to_tensor(gt_)

        gt_ = gt_.unsqueeze(0)

        return im_, gt_

    def val_transform(self, im, gt):
        # im = np.array(im).astype(np.float32)
        # gt = np.array(gt).astype(np.float32)

        # Upper_cor = random.randint(1, 464)
        # Left_cor = random.randint(1, 800)
        Upper_cor = random.randint(1, 360)
        Left_cor = random.randint(1, 640)
        transform = my_transforms.Compose([
            my_transforms.Crop(Upper_cor, Left_cor, 360, 640),
            # my_transforms.Crop(Upper_cor, Left_cor, 256, 480),
            # my_transforms.Resize(460 / 240, interpolation='bilinear'),
            # my_transforms.CenterCrop(self.size)
        ])

        im_ = transform(im)
        gt_ = transform(gt)

        im_ = np.array(im_).astype(np.float32)
        gt_ = np.array(gt_).astype(np.float32)

        im_ /= 255.0
        gt_ /= 1000.0
        im_ = to_tensor(im_)
        gt_ = to_tensor(gt_)

        gt_ = gt_.unsqueeze(0)
        return im_, gt_

    # def train_transform(self, rgb, depth):
    #     s = np.random.uniform(1.0, 1.5)  # random scaling
    #     depth_np = depth / s
    #     angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
    #     do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
    #
    #     # perform 1st step of data augmentation
    #     transform = transforms.Compose([
    #         transforms.Resize(288.0 / iheight),  # this is for computational efficiency, since rotation can be slow
    #         transforms.Rotate(angle),
    #         transforms.Resize(s),
    #         transforms.CenterCrop(self.output_size),
    #         transforms.HorizontalFlip(do_flip)
    #     ])
    #     # rgb_np = transform(rgb)
    #     # rgb_np = self.color_jitter(rgb_np)  # random color jittering
    #     # rgb_np = np.asfarray(rgb_np, dtype='float') / 255
    #     rgb_np = np.asfarray(rgb, dtype='float') / 255
    #     depth_np = transform(depth_np)
    #
    #     return rgb_np, depth_np
    #
    # def val_transform(self, rgb, depth):
    #     depth_np = depth
    #     transform = transforms.Compose([
    #         transforms.Resize(288.0 / iheight),
    #         transforms.CenterCrop(self.output_size),
    #     ])
    #     rgb_np = transform(rgb)
    #     rgb_np = np.asfarray(rgb_np, dtype='float') / 255
    #     depth_np = transform(depth_np)
    #
    #     return rgb_np, depth_np


# array to tensor
from dataloaders import transforms as my_transforms

to_tensor = my_transforms.ToTensor()


class HackerFolder(Dataset):
    """
    root/
        train: xxx/xxx.jpg, xxxx/xxxx.png
        val: xxx/xxx.jpg, xxxx/xxxx.png
        test.txt
        val.txt
    """

    def __init__(self, root, data, transform=None):
        # data = 'test.txt' or 'val.txt'
        with open(os.path.join(root, data), 'r') as f:
            lines = f.readlines()
        self.root = root
        self.data = lines
        self.transform = transform
        self.loader = pil_loader
        self.size = (1280, 720)

    def __getitem__(self, idx):
        sample = self.data[idx].strip('\n').split()
        # im_path, gt_path
        im_path = os.path.join(self.root, sample[0])
        gt_path = os.path.join(self.root, sample[1])
        im = self.loader(im_path)
        gt = self.loader(gt_path, rgb=False)

        if self.data == 'train.txt':
            im, gt = self.train_transform(im, gt)
        else:
            im, gt = self.val_transform(im, gt)

        return im, gt

    def __len__(self):
        return len(self.data)

    def train_transform(self, im, gt):
        im = np.array(im).astype(np.float32)
        gt = np.array(gt).astype(np.float32)

        s = np.random.uniform(1.0, 1.5)  # random scaling
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
        color_jitter = my_transforms.ColorJitter(0.4, 0.4, 0.4)

        transform = my_transforms.Compose([
            my_transforms.Crop(130, 10, 240, 1200),
            my_transforms.Resize(460 / 240, interpolation='bilinear'),
            my_transforms.Rotate(angle),
            my_transforms.Resize(s),
            my_transforms.CenterCrop(self.size),
            my_transforms.HorizontalFlip(do_flip)
        ])

        im_ = transform(im)
        im_ = color_jitter(im_)

        gt_ = transform(gt)

        im_ = np.array(im_).astype(np.float32)
        gt_ = np.array(gt_).astype(np.float32)

        im_ /= 255.0
        gt_ /= 100.0 * s
        im_ = to_tensor(im_)
        gt_ = to_tensor(gt_)

        gt_ = gt_.unsqueeze(0)

        return im_, gt_

    def val_transform(self, im, gt):
        im = np.array(im).astype(np.float32)
        gt = np.array(gt).astype(np.float32)

        transform = my_transforms.Compose([
            my_transforms.Crop(130, 10, 240, 1200),
            my_transforms.Resize(460 / 240, interpolation='bilinear'),
            my_transforms.CenterCrop(self.size)
        ])

        im_ = transform(im)
        gt_ = transform(gt)

        im_ = np.array(im_).astype(np.float32)
        gt_ = np.array(gt_).astype(np.float32)

        im_ /= 255.0
        gt_ /= 100.0
        im_ = to_tensor(im_)
        gt_ = to_tensor(gt_)

        gt_ = gt_.unsqueeze(0)
        return im_, gt_


if __name__ == '__main__':
    from tqdm import tqdm

    test = HackerDataloader("/home/dahai/hackthon/dataset", type='val')
    bug_test = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False,
                                           num_workers=1, pin_memory=True)
    for im, gt in tqdm(bug_test):
        # print(im)

        valid = (gt > 0.0)
        # print(torch.max(gt[valid]), torch.min(gt[valid]))
        # print(gt.size())
        # print(im.size())
