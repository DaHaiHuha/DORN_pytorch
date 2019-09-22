# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/27 19:28
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

from network import DORN_nyu, DORN_kitti, DORN_hacker


def get_models(dataset='nyu', pretrained=True, freeze=True):  # TODO false freeze
    if dataset == 'nyu':
        return DORN_nyu.DORN(pretrained=pretrained, freeze=freeze)
    elif dataset == 'kitti':
        return DORN_kitti.DORN(pretrained=pretrained, freeze=freeze)
    elif dataset == 'hacker':
        print('Using NYU DORN like net structure for hacker!')
        return DORN_hacker.DORN(pretrained=pretrained, freeze=freeze)
    else:
        print('no model based on dataset-', dataset)
        exit(-1)
