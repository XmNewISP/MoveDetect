#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import torchvision.transforms.functional as tvF
import torch.nn.functional as F
from unet import UNet, MdNet
from utils import *

import os
import json

class MoveDetect(object):
    """Implementation of MoveDetect """

    def __init__(self, params, trainable):
        """Initializes model."""

        self.p = params
        self.trainable = trainable
        self._compile()

    def _compile(self):
        """Compiles model (architecture, loss function, optimizers, etc.)."""

        print('MoveDetect: Learning Image Restoration without Clean Data (Lethinen et al., 2018)')

        # Model
        self.model = MdNet(in_channels=6,out_channels=2)

        # Set optimizer and loss, if in training mode
        if self.trainable:
            self.optim = Adam(self.model.parameters(),
                              lr=self.p.learning_rate,
                              betas=self.p.adam[:2],
                              eps=self.p.adam[2])

            # Learning rate adjustment
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim,
                patience=self.p.nb_epochs/10, factor=0.5, verbose=True)

            # Loss function
            self.loss = nn.CrossEntropyLoss()

        # CUDA support
        self.use_cuda = torch.cuda.is_available() and self.p.cuda
        if self.use_cuda:
            self.model = self.model.cuda()
            if self.trainable:
                self.loss = self.loss.cuda()

    def _print_params(self):
        """Formats parameters to print when training."""

        print('Training parameters: ')
        self.p.cuda = self.use_cuda
        param_dict = vars(self.p)
        pretty = lambda x: x.replace('_', ' ').capitalize()
        print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        print()

    def save_model(self, epoch, stats, first=False):
        """Saves model to files; can be overwritten at every epoch to save disk space."""

        # Create directory for model checkpoints, if nonexistent
        if first:
            ckpt_dir_name = "movedetect"
            self.ckpt_dir = os.path.join(self.p.ckpt_save_path, ckpt_dir_name)
            if not os.path.isdir(self.p.ckpt_save_path):
                os.mkdir(self.p.ckpt_save_path)
            if not os.path.isdir(self.ckpt_dir):
                os.mkdir(self.ckpt_dir)

        # Save checkpoint dictionary
        if self.p.ckpt_overwrite:
            fname_unet = '{}/md.pt'.format(self.ckpt_dir)
        else:
            valid_loss = stats['valid_loss'][epoch]
            fname_unet = '{}/md-epoch{}-{:>1.5f}.pt'.format(self.ckpt_dir, epoch + 1, valid_loss)
        print('Saving checkpoint to: {}\n'.format(fname_unet))
        torch.save(self.model.state_dict(), fname_unet)

        # Save stats to JSON
        fname_dict = '{}/n2n-stats.json'.format(self.ckpt_dir)
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)

    def load_model(self, ckpt_fname):
        """Loads model from checkpoint file."""

        print('Loading checkpoint from: {}'.format(ckpt_fname))
        if self.use_cuda:
            self.model.load_state_dict(torch.load(ckpt_fname))
        else:
            self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))

    def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader):
        """Tracks and saves starts after each epoch."""

        # Evaluate model on validation set
        print('\rTesting model on validation set... ', end='')
        epoch_time = time_elapsed_since(epoch_start)[0]
        valid_loss, valid_time = self.eval(valid_loader)
        show_on_epoch_end(epoch_time, valid_time, valid_loss)

        # Decrease learning rate if plateau
        self.scheduler.step(valid_loss)

        # Save checkpoint
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        self.save_model(epoch, stats, epoch == 0)

        # Plot stats
        if self.p.plot_stats:
            loss_str = f'loss'
            plot_per_epoch(self.ckpt_dir, 'Valid loss', stats['valid_loss'], loss_str)

    def eval(self, valid_loader):
        """Evaluates denoiser on validation set."""

        self.model.train(False)

        valid_start = datetime.now()
        loss_meter = AvgMeter()

        for batch_idx, (source, target) in enumerate(valid_loader):
            if self.use_cuda:
                source = source.cuda()
                target = target.cuda()

            # MoveDetect
            source_md = self.model(source)

            # Update loss
            loss = self.loss(source_md,target)
            loss_meter.update(loss.item())

        valid_loss = loss_meter.avg
        valid_time = time_elapsed_since(valid_start)[0]
        return valid_loss, valid_time

    def train(self, train_loader, valid_loader):
        """Trains denoiser on training set."""

        self.model.train(True)

        self._print_params()
        num_batches = len(train_loader)
        if self.p.report_interval == 0:
            self.p.report_interval = num_batches
        print("--------->num_batches:",num_batches)
        print("--------->report_interval:",self.p.report_interval)  
        assert num_batches % self.p.report_interval == 0, 'Report interval must divide total number of batches'

        # Dictionaries of tracked stats
        stats = {'train_loss': [],
                 'valid_loss': []}

        # Main training loop
        train_start = datetime.now()
        for epoch in range(self.p.nb_epochs):
            print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))

            # Some stats trackers
            epoch_start = datetime.now()
            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()

            # Minibatch SGD
            for batch_idx, (source, target) in enumerate(train_loader):
                batch_start = datetime.now()
                progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)

                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()         
                # MoveDetect image
                source_md = self.model(source)
                loss = self.loss(source_md, target)
                loss_meter.update(loss.item())

                # Zero gradients, perform a backward pass, and update the weights
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # Report/update statistics
                time_meter.update(time_elapsed_since(batch_start)[1])
                if (batch_idx + 1) % self.p.report_interval == 0 and batch_idx:
                    show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
                    train_loss_meter.update(loss_meter.avg)
                    loss_meter.reset()
                    time_meter.reset()

            # Epoch end, save and reset tracker
            self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_loader)
            train_loss_meter.reset()

        train_elapsed = time_elapsed_since(train_start)[0]

    #ÕûÍ¼Êä³ö
    def test2(self):
        """Evaluates denoiser on test set."""

        self.model.train(False)

        # Create directory for denoised images
        movedetect_dir = os.path.dirname(self.p.data)
        save_path = os.path.join(movedetect_dir, 'movedetect')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        namelist = os.listdir(movedetect_dir)
        filelist=[os.path.join(movedetect_dir,name) for name in namelist if ("movedetect" not in name and "_other" not in name)  ]
        print(filelist)
        # Load PIL image
        for img_path in filelist:
            img =  Image.open(img_path).convert('RGB')
            w, h = img.size
            '''
            if w % 32 != 0:
                w = (w//32)*32
            if h % 32 != 0:
                h = (h//32)*32
            '''
            img = tvF.crop(img, 0, 0, h, w)
            source = tvF.to_tensor(img)
            source = source.unsqueeze(0)
            print(source.size())
            other_path = img_path.replace(".jpg", "_other.jpg")
            other = Image.open(other_path).convert('RGB')
            other = tvF.crop(other, 0, 0, h, w)
            other = tvF.to_tensor(other)
            other = other.unsqueeze(0)
            print(other.size())
            
            if self.use_cuda:
                source = source.cuda()
                other = other.cuda()

            # MoveDetect
            input = torch.cat((source,other),dim=1)
            output = self.model(input)
            #print("111--------->",output.size())
            #print(output)            
            result = F.softmax(output, dim=1).squeeze(0).detach().cpu()
            #print("222--------->",result.size())
            #print(result[1])           
            md_image = tvF.to_pil_image(result[1])
            fname = os.path.basename(img_path)
            md_image.save(os.path.join(save_path, f'{fname}-md.png'))

            
