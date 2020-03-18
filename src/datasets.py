#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset, DataLoader

import os
from sys import platform
import numpy as np
import random
from string import ascii_letters
from PIL import Image, ImageFont, ImageDraw
import OpenEXR


from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import threading
from torchvision import transforms as transforms

def load_dataset(root_dir, params, workers=0,shuffled=False,single=False):
    """Loads dataset and returns corresponding data loader."""

    #dataset = MoveDetectDataset(root_dir,params.crop_size)
    dataset = MoveDetectDataset2(root_dir,params.crop_size)

    # Use batch size of 1, if requested (e.g. test set)
    if single:
        return DataLoader(dataset, batch_size=1, shuffle=shuffled,drop_last=True)
    else:
        return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled,num_workers=workers,drop_last=True)

class AbstractDataset(Dataset):
    """Abstract dataset class for Noise2Noise."""

    def __init__(self, root_dir, crop_size=128):
        """Initializes abstract dataset."""

        super(AbstractDataset, self).__init__()

        self.imgs = []
        self.root_dir = root_dir
        self.crop_size = crop_size

    def _random_crop(self, img_list):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """

        w, h = img_list[0].size
        assert w >= self.crop_size and h >= self.crop_size, \
            f'Error: Crop size: {self.crop_size}, Image size: ({w}, {h})'
        cropped_imgs = []
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)

        for img in img_list:
            # Resize if dimensions are too small
            if min(w, h) < self.crop_size:
                img = tvF.resize(img, (self.crop_size, self.crop_size))

            # Random crop
            cropped_imgs.append(tvF.crop(img, i, j, self.crop_size, self.crop_size))

        return cropped_imgs


    def __getitem__(self, index):
        """Retrieves image from data folder."""

        raise NotImplementedError('Abstract method not implemented!')


    def __len__(self):
        """Returns length of dataset."""
        print("Dataset--------->len=",len(self.imgs))
        return len(self.imgs)

#geyijun@2020-03-13
#用真实的图像训练;同时支持从子目录中去取图。
#不同用例中去出来的为运动图，相同用例中静止图
class MoveDetectDataset(AbstractDataset):
    """Class for injecting random noise into dataset."""
    """A dataset for loading filelist as.
    like::
        train/case1_lv1/1.jpg
        train/case1_lv1/N.jpg
        train/case1_lv2/1.jpg
        train/case1_lv2/N.jpg
        train/case2_lv1/1.jpg
        train/case2_lv1/N.jpg
        train/case2_lv2/1.jpg
        train/case2_lv2/N.jpg
    """
    def __init__(self, root_dir, crop_size):
        """Initializes noisy image dataset."""
        super(MoveDetectDataset, self).__init__(root_dir, crop_size)
        #列出所有的场景目录的文件列表
        self.imgs = []
        self.case_indexmap = {}    #casename->(begin,end)
        caselist = os.listdir(root_dir)
        assert len(caselist)>=2, f"Case Num is less than two {root_dir}!!!"
        begin_index = 0
        for case in caselist:
            namelist = os.listdir(os.path.join(root_dir,case))
            filelist=[os.path.join(root_dir,case,name) for name in namelist if (name != "groundtruth.jpg")]
            self.imgs.extend(filelist)
            self.case_indexmap[os.path.join(root_dir,case)] = (begin_index,begin_index+len(filelist))
            begin_index += len(filelist)
        print("MoveDetectDataset,imgs=",len(self.imgs))
        print("MoveDetectDataset,case_indexmap=",self.case_indexmap)
        
    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""
        
        # Load PIL image
        img_path = self.imgs[index]
        img =  Image.open(img_path).convert('RGB')

        #取同源的和异源的组合在一起，构造other_img
        begin_index,end_index = self.case_indexmap[os.path.split(img_path)[0]]
        for i in range(100): 
            same_index = np.random.randint(begin_index,end_index)
            if(same_index != index):
                break
        for i in range(100):
            diff_index = np.random.randint(len(self.imgs))
            if(diff_index < begin_index) or (diff_index >= end_index):
                break
        same_path = self.imgs[same_index]
        same = Image.open(same_path).convert('RGB')
        diff_path = self.imgs[diff_index]
        diff = Image.open(diff_path).convert('RGB')
        
        # Random square crop
        if self.crop_size != 0:
            crops_imgs = self._random_crop([img,same,diff])
            img_crop = crops_imgs[0]
            same_crop = crops_imgs[1] 
            diff_crop = crops_imgs[2]
            #crops_imgs[0].save('crop0.png')
            #crops_imgs[1].save('crop1.png')
            #crops_imgs[2].save('crop2.png')
            #print("------move_label:",move_label)
            #exit()

        #非规则性合并
        #参考:https://blog.csdn.net/leemboy/article/details/83792729
        #创建numpy
        mask_np = np.zeros(same_crop.size,dtype=int)
        mask_np[same_crop.size[0]//2:,] = 255               #一半黑一半白
        mask= Image.fromarray(np.uint8(mask_np))            #转成Image
        mask = mask.rotate(np.random.randint(-180,180))     #随机旋转
        other = Image.composite(diff_crop,same_crop,mask)   #组合在一起
        '''
        print(tvF.to_tensor(mask).size())
        print(tvF.to_tensor(mask)[0,0,:])
        mask.save('mask.png')
        img_crop.save('img_crop.png')
        same_crop.save('same_crop.png')
        diff_crop.save('diff_crop.png')        
        other.save('other.png')  
        print("------>:write ok")
        exit()
        '''

        #concat在一起
        t0 = tvF.to_tensor(img_crop)
        t1 = tvF.to_tensor(other)
        source = torch.cat((t0,t1),dim=0)
        target = tvF.to_tensor(mask).squeeze()
        #############################################
        #注意:target不需要channel(class) 这个维度
        #https://blog.csdn.net/qq_31347869/article/details/104074421
        #############################################
        #print("source.size=",source.size())
        #print("target.size=",target.size())
        return source, target.long()


#geyijun@2020-03-17
#一次取一个感受野的大小,为了避免label的块状效应
class MoveDetectDataset2(AbstractDataset):
    """Class for injecting random noise into dataset."""
    """A dataset for loading filelist as.
    like::
        train/case1_lv1/1.jpg
        train/case1_lv1/N.jpg
        train/case1_lv2/1.jpg
        train/case1_lv2/N.jpg
        train/case2_lv1/1.jpg
        train/case2_lv1/N.jpg
        train/case2_lv2/1.jpg
        train/case2_lv2/N.jpg
    """
    def __init__(self, root_dir, crop_size):
        """Initializes noisy image dataset."""
        super(MoveDetectDataset2, self).__init__(root_dir, crop_size)
        #列出所有的场景目录的文件列表
        self.crop_size = 32         #网络设计绑定为32
        self.imgs = []
        self.case_level_indexmap = {}    #case_level->(begin,end)
        case_level_list = os.listdir(root_dir)
        begin_index = 0
        for case_level in case_level_list:
            namelist = os.listdir(os.path.join(root_dir,case_level))
            filelist=[os.path.join(root_dir,case_level,name) for name in namelist if (name != "groundtruth.jpg")]
            self.imgs.extend(filelist)
            self.case_level_indexmap[os.path.join(root_dir,case_level)] = (begin_index,begin_index+len(filelist))
            begin_index += len(filelist)
        random.shuffle(self.imgs)
        print("MoveDetectDataset,imgs=",len(self.imgs))
        print("MoveDetectDataset,case_level_indexmap=",self.case_level_indexmap)

        #为了提升效率，在一张图片打开之后，多crop几个
        self.mutex = threading.Lock()
        self.list_cache_crops = list()

    def __len__(self):
        """Returns length of dataset."""
        print("Dataset--------->len=",len(self.imgs)*100)
        return len(self.imgs)*100

    #一次多crop一些
    def do_more_crops(self, index): 
        # Load PIL image
        img_path = self.imgs[index]
        img =  Image.open(img_path).convert('RGB')
        #############################################
        label = np.random.choice(a=[0,1], size=1, replace=False, p=[0.1,0.9])  #正样本概率更高一些
        if (label==1) :
            for i in range(100): 
                random_index = np.random.randint(len(self.imgs)) 
                random_img_path = self.imgs[random_index]
                if (img_path.split("/")[-2].split("_")[0] != random_img_path.split("/")[-2].split("_")[0]):
                    break
                if i >= 99:
                    assert False, f"Get random img path Error!!!"
        else :
            begin_index,end_index = self.case_level_indexmap[os.path.split(img_path)[0]]
            random_index = np.random.randint(end_index-begin_index)+1    #编号从1开始
            random_img_path = os.path.split(img_path)[0]+"/"+"{:0>5d}.jpg".format(random_index)
        #print(label,img_path,random_img_path)
        #print(img_path.split("/")[-2].split("_")[0])
        #print(random_img_path.split("/")[-2].split("_")[0])
        other = Image.open(random_img_path).convert('RGB')        
        #############################################
        crops_list = list()
        #一次crop个
        for i in range(256):
            crops_imgs = self._random_crop([img,other])
            img_crop = crops_imgs[0]
            other_crop = crops_imgs[1]
            #crops_imgs[0].save('crop0.png')
            #crops_imgs[1].save('crop1.png')
            #print("------label:",label)
            #exit()
            crops_list.append((img_crop,other_crop,label))
        return crops_list
        
    def is_cache_list_empty(self):
        self.mutex.acquire()  
        ret = len(self.list_cache_crops) <= 0
        self.mutex.release()
        return ret
    def get_from_cache_list(self):
        self.mutex.acquire()  
        item = self.list_cache_crops.pop()
        self.mutex.release()
        return item
    def put_to_cache_list(self,crops):
        self.mutex.acquire()  
        self.list_cache_crops.extend(crops)
        self.mutex.release()

    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""        
        #先判断缓冲队列
        if self.is_cache_list_empty():
            crops = self.do_more_crops(index%len(self.imgs))
            self.put_to_cache_list(crops)

        item = self.get_from_cache_list()
        img_crop = item[0]
        other_crop = item[1]
        label  = item[2]
        #concat在一起
        t0 = tvF.to_tensor(img_crop)
        t1 = tvF.to_tensor(other_crop)
        source = torch.cat((t0,t1),dim=0)
        target_np = np.array([[label]],dtype=int)
        target = tvF.to_tensor(target_np).squeeze(0)
        #print(target.size())
        return source, target.long()

