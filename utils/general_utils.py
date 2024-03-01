#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import struct
import os

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def vis_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET, constant_max=120):
    """
    depth: (H, W)
    """
    depthmap = np.nan_to_num(depth) # change nan to 0

    # x_ = (255 - x)[:,:,None].repeat(3,axis=-1)

    # x = x[]
    # threshold
    # constant_max = np.percentile(depthmap, 90)
    depthmap_valid_count = (depthmap < 300).sum()
    constant_max = np.percentile(depthmap[depthmap<300], 99) if depthmap_valid_count > 10 else 60
    # constant_max = 1
    # constant_min = 0
    constant_min = np.percentile(depthmap, 1) if np.percentile(depthmap, 1) < constant_max else 0
    normalizer = mpl.colors.Normalize(vmin=constant_min, vmax=constant_max)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
    depth_vis_color = (mapper.to_rgba(depthmap)[:, :, :3] * 255).astype(np.uint8)
    # all_white = np.ones_like(x_) * 255
    # x_ = x_ * (1-bg_mask)[:,:,None] + all_white * bg_mask[:,:,None]
    # x_ = x_.astype(np.uint8)
    # x_ = cv2.cvtColor(x_, cv2.COLOR_BGR2RGB)
    return depth_vis_color, [np.percentile(depthmap, 0), np.percentile(depthmap, 99)]

def vis_depth1(depth, minmax=None, cmap=cv2.COLORMAP_JET, constant_max=120):
    """
    depth: (H, W)
    """
    depthmap = np.nan_to_num(depth) # change nan to 0

    # x_ = (255 - x)[:,:,None].repeat(3,axis=-1)

    # x = x[]
    # threshold
    # constant_max = np.percentile(depthmap, 90)
    depthmap_valid_count = (depthmap < 300).sum()
    # constant_max = np.percentile(depthmap[depthmap<300], 99) if depthmap_valid_count > 10 else 60
    constant_max = 10
    constant_min = 0.5
    # constant_min = np.percentile(depthmap, 1) if np.percentile(depthmap, 1) < constant_max else 0
    normalizer = mpl.colors.Normalize(vmin=constant_min, vmax=constant_max)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
    depth_vis_color = (mapper.to_rgba(depthmap)[:, :, :3] * 255).astype(np.uint8)
    # all_white = np.ones_like(x_) * 255
    # x_ = x_ * (1-bg_mask)[:,:,None] + all_white * bg_mask[:,:,None]
    # x_ = x_.astype(np.uint8)
    # x_ = cv2.cvtColor(x_, cv2.COLOR_BGR2RGB)
    return depth_vis_color, [np.percentile(depthmap, 0), np.percentile(depthmap, 99)]

def readDepthDmb(file_path):
    inimage = open(file_path, "rb")
    if not inimage:
        print("Error opening file", file_path)
        return -1

    type = -1

    type = struct.unpack("i", inimage.read(4))[0]
    h = struct.unpack("i", inimage.read(4))[0]
    w = struct.unpack("i", inimage.read(4))[0]
    nb = struct.unpack("i", inimage.read(4))[0]

    if type != 1:
        inimage.close()
        return -1

    dataSize = h * w * nb

    depth = np.zeros((h, w), dtype=np.float32)
    depth_data = np.frombuffer(inimage.read(dataSize * 4), dtype=np.float32)
    depth_data = depth_data.reshape((h, w))
    np.copyto(depth, depth_data)

    inimage.close()
    return depth

def readNormalDmb(file_path):
    try:
        with open(file_path, 'rb') as inimage:
            type = np.fromfile(inimage, dtype=np.int32, count=1)[0]
            h = np.fromfile(inimage, dtype=np.int32, count=1)[0]
            w = np.fromfile(inimage, dtype=np.int32, count=1)[0]
            nb = np.fromfile(inimage, dtype=np.int32, count=1)[0]

            if type != 1:
                print("Error: Invalid file type")
                return -1

            dataSize = h * w * nb

            normal = np.zeros((h, w, 3), dtype=np.float32)
            normal_data = np.fromfile(inimage, dtype=np.float32, count=dataSize)
            normal_data = normal_data.reshape((h, w, nb))
            normal[:, :, :] = normal_data[:, :, :3]

            return normal

    except IOError:
        print("Error opening file", file_path)
        return -1

def read_propagted_depth(path):    
    cost = readDepthDmb(os.path.join(path, 'costs.dmb'))
    cost[cost==np.nan] = 2
    cost[cost < 0] = 2
    # mask = cost > 0.5

    depth = readDepthDmb(os.path.join(path, 'depths.dmb'))
    # depth[mask] = 300
    depth[np.isnan(depth)] = 300
    depth[depth < 0] = 300
    depth[depth > 300] = 300
    
    normal = readNormalDmb(os.path.join(path, 'normals.dmb'))

    return depth, cost, normal

def load_pairs_relation(path):
    pairs_relation = []
    num = 0
    with open(path, 'r') as file:
        num_images = int(file.readline())
        for i in range(num_images):

            ref_image_id = int(file.readline())
            if i != ref_image_id:
                print(ref_image_id)
                print(i)

            src_images_infos = file.readline().split()
            num_src_images = int(src_images_infos[0])
            src_images_infos = src_images_infos[1:]
            
            pairs = []
            #only fetch the first 4 src images
            for j in range(num_src_images):
                id, score = int(src_images_infos[2*j]), int(src_images_infos[2*j+1])
                #the idx needs to align to the training images
                if score <= 0.0 or id % 8 == 0:
                    continue
                id = (id // 8) * 7 + (id % 8) - 1
                pairs.append(id)
                
                if len(pairs) > 3:
                    break
                
            if ref_image_id % 8 != 0:
                #only load the training images
                pairs_relation.append(pairs)
            else:
                num = num + 1
            
    return pairs_relation