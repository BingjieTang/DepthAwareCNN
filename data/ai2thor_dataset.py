import os.path
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
import h5py
from data.base_dataset import *
from PIL import Image
import math, random
import time
import h5py

def make_dataset_from_hdf5(hdf5filename):
    images = []
    depths = []

    # images_filename, depth_filename = hdf5filename.split()
    # rgb_path = "../FloorPlan1_physics/images.hdf5"
    # depth_path = "../FloorPlan1_physics/depth.hdf5"

    rgb_data = h5py.File(hdf5filename, "r")
    # depth_data = h5py.File(depth_filename, "r")

    rgb_data_keys = list(rgb_data.keys())
    # depth_data_keys = list(depth_data.keys())
    # assert len(rgb_data_keys) == len(depth_data_keys)

    # for key in rgb_data_keys:
    #     images.append(rgb_data[key])
    #     depths.append(depth_data[key])
    # print(len(images))
    # assert len(rgb_data_keys) == len(depth_data_keys)
    return {'keys': rgb_data_keys}

class Ai2ThorDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        np.random.seed(int(time.time()))
        self.paths_dict = make_dataset_from_hdf5(opt.list)
        self.len = len(self.paths_dict['keys'])
        # self.label_weight = torch.Tensor(label_weight)
        self.datafile = 'ai2thor_dataset.py'

    def __getitem__(self, index):
        key = self.paths_dict['keys'][index]

        rgb_path = "../FloorPlan1_physics/images.hdf5"
        depth_path = "../FloorPlan1_physics/depth.hdf5"

        rgb_data = h5py.File(rgb_path, "r")
        depth_data = h5py.File(depth_path, "r")

        
        img = np.asarray(rgb_data[key]) #.astype(np.uint8)
        depth = np.asarray(depth_data[key]).astype(np.float32)/120
        # depth = np.asarray(Image.open(self.paths_dict['depths'][index])).astype(np.float32)/120. # 1/10 * depth
        # key = np.asanyarray(self.paths_dict['keys'][index])
        # seg = np.asarray(Image.open(self.paths_dict['segs'][index]))-1

        params = get_params_sunrgbd(self.opt, img.shape[:2], maxcrop=0.7, maxscale=1.1)
        depth_tensor_tranformed = transform(depth, params, normalize=False,istrain=self.opt.isTrain)
        # seg_tensor_tranformed = transform(seg, params, normalize=False,method='nearest',istrain=self.opt.isTrain)
        if self.opt.inputmode == 'bgr-mean':
            img_tensor_tranformed = transform(img, params, normalize=False, istrain=self.opt.isTrain, option=1)
        else:
            img_tensor_tranformed = transform(img, params, istrain=self.opt.isTrain, option=1)

        return {'image': img_tensor_tranformed,
                'depth': depth_tensor_tranformed,
                'key': key
                }
                # 'seg': seg_tensor_tranformed,
                # 'imgpath': self.paths_dict['segs'][index]}

    def __len__(self):
        return self.len

    def name(self):
        return 'ai2thor_dataset'


class Ai2ThorDataset_val(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        np.random.seed(8934)
        self.paths_dict = make_dataset_from_hdf5(opt.list)
        self.len = len(self.paths_dict['images'])
        # self.label_weight = torch.Tensor(label_weight)
        # self.datafile = 'ai2thor_dataset.py'

    def __getitem__(self, index):

        img = np.asarray(Image.open(self.paths_dict['images'][index])) #.astype(np.uint8)
        depth = np.asarray(Image.open(self.paths_dict['depths'][index])).astype(np.float32)/120. # 1/10 * depth
        key = np.asanyarray(self.paths_dict['keys'][index])
        # seg = np.asarray(Image.open(self.paths_dict['segs'][index]))-1

        params = get_params_sunrgbd(self.opt, img.shape, maxcrop=0.7, maxscale=1.1)
        depth_tensor_tranformed = transform(depth, params, normalize=False,istrain=self.opt.isTrain)
        # seg_tensor_tranformed = transform(seg, params, normalize=False,method='nearest',istrain=self.opt.isTrain)
        if self.opt.inputmode == 'bgr-mean':
            img_tensor_tranformed = transform(img, params, normalize=False, istrain=self.opt.isTrain, option=1)
        else:
            img_tensor_tranformed = transform(img, params, istrain=self.opt.isTrain, option=1)

        return {'image': img_tensor_tranformed,
                'depth': depth_tensor_tranformed,
                'key': key
                }
                # 'seg': seg_tensor_tranformed,
                # 'imgpath': self.paths_dict['segs'][index]}

    def __len__(self):
        return self.len

    def name(self):
        return 'ai2thor_dataset'
