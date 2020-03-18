#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from datasets import load_dataset
from movedetect import MoveDetect
from argparse import ArgumentParser

def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of MoveDetect ')

    # Data parameters
    parser.add_argument('-t', '--train-dir', help='training set path', default='./../data/train')
    parser.add_argument('-v', '--valid-dir', help='test set path', default='./../data/valid')
    parser.add_argument('--resume-ckpt', help='resume model checkpoint',default=None, type=str)
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./../ckpts')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--report-interval', help='batch report interval', default=0, type=int)
    parser.add_argument('--workers-num', help='workers-num', default=0, type=int)

    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=4, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=100, type=int)
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true')

    # Corruption parameters
    parser.add_argument('-c', '--crop-size', help='random crop size', default=128, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    """Trains MoveDetect."""

    # Parse training parameters
    params = parse_args()

    # Train/valid datasets
    train_loader = load_dataset(params.train_dir, params, workers=params.workers_num,shuffled=True)
    valid_loader = load_dataset(params.valid_dir, params, workers=params.workers_num,shuffled=False)

    # Initialize model and train
    md = MoveDetect(params, trainable=True)
    if params.resume_ckpt:
        md.load_model(params.resume_ckpt)
    md.train(train_loader, valid_loader)
