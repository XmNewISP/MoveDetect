#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

#�����������
#ȥ��unsample���̣�ͬʱȥ��pad��������ƹ̶�����Ұ��С�����1x1�ĳ߶ȣ�����label�ȽϺ���ơ�
#����Ұ���Ϊ32*32
class MdNet(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""
    
    def __init__(self, in_channels=6, out_channels=2):
        """Initializes U-Net."""

        super(MdNet, self).__init__()
        #
        self.net = nn.Sequential(               #32
            nn.Conv2d(in_channels, 32, 3),      #30
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    #15
            nn.Conv2d(32, 64, 2),               #14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    #7
            nn.Conv2d(64, 128, 2),              #6
            nn.ReLU(inplace=True),      
            nn.MaxPool2d(2),                    #3
            nn.Conv2d(128, 128, 2),             #2
            nn.ReLU(inplace=True),              
            nn.Conv2d(128, out_channels, 2))    #1
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """
        return self.net(x)

