import sys
import copy
from glob import glob
import math
import os

import torch

import nvvl

class NVVL():
    def __init__(self, frames, is_cropped, crop_size, root, 
                 batchsize=1, device_id=0, 
                 shuffle=False, distributed=False, fp16=False):
        self.root = root
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.distributed = distributed
        self.frames = frames
        self.device_id = device_id

        self.is_cropped = is_cropped
        self.crop_size = crop_size

        self.files = glob(os.path.join(self.root, '*.mp4'))[64:66]

        if len(self.files) < 1:
            print(("[Error] No video files in %s" % (self.root)))
            raise LookupError

        if fp16:
            tensor_type = 'half'
        else:
            tensor_type = 'float'
            
        self.image_shape = nvvl.video_size_from_file(self.files[0])
        
        height = min(self.image_shape.height, self.crop_size[0])
        width = min(self.image_shape.width, self.crop_size[1])
        
        processing = {"input": nvvl.ProcessDesc(type=tensor_type, height=height, width=width,
                                               random_crop=self.is_cropped, random_flip=False,
                                               normalized=True, color_space="RGB", dimension_order="cfhw")}
        
        dataset = nvvl.VideoDataset(self.files,
                                   sequence_length=self.frames,
                                   device_id=self.device_id,
                                   processing=processing)
        
        self.loader = nvvl.VideoLoader(dataset, batch_size=self.batchsize, shuffle=self.shuffle, distributed=self.distributed)
        
    def __len__(self):
        return len(self.loader)
    
    def __iter__(self):
        return iter(self.loader)