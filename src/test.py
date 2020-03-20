#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from datasets import load_dataset
from movedetect import MoveDetect

from argparse import ArgumentParser


def parse_args():
    """Command-line argument parser for testing."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of MoveDetect from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('-d', '--data', help='dataset root path', default='../data')
    parser.add_argument('--load-ckpt', help='load model checkpoint')
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    return parser.parse_args()
        
if __name__ == '__main__':
    """Tests Noise2Noise."""

    # Parse test parameters
    params = parse_args()

    # Initialize model and test
    md = MoveDetect(params, trainable=False)
    md.load_model(params.load_ckpt)
    md.test2()

