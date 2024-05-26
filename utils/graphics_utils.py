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
import math
import numpy as np
from typing import NamedTuple
import cv2
import os
from gaussianpro import propagate

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def write_cam_txt(cam_path, K, w2c, depth_range):
    with open(cam_path, "w") as file:
        file.write("extrinsic\n")
        for row in w2c:
            file.write(" ".join(str(element) for element in row))
            file.write("\n")

        file.write("\nintrinsic\n")
        for row in K:
            file.write(" ".join(str(element) for element in row))
            file.write("\n")
        
        file.write("\n")
        
        file.write(" ".join(str(element) for element in depth_range))
        file.write("\n")

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def init_image_coord(height, width):
    x_row = np.arange(0, width)
    x = np.tile(x_row, (height, 1))
    x = x[np.newaxis, :, :]
    x = x.astype(np.float32)
    x = torch.from_numpy(x.copy()).cuda()
    u_u0 = x - width/2.0

    y_col = np.arange(0, height)  # y_col = np.arange(0, height)
    y = np.tile(y_col, (width, 1)).T
    y = y[np.newaxis, :, :]
    y = y.astype(np.float32)
    y = torch.from_numpy(y.copy()).cuda()
    v_v0 = y - height/2.0
    return u_u0, v_v0

def depth_to_xyz(depth, intrinsic):
    b, c, h, w = depth.shape
    u_u0, v_v0 = init_image_coord(h, w)
    x = (u_u0 - intrinsic[0][2]) * depth / intrinsic[0][0]
    y = (v_v0 - intrinsic[1][2]) * depth / intrinsic[1][1]
    z = depth
    pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1) # [b, h, w, c]
    return pw

def get_surface_normalv2(xyz, patch_size=5):
    """
    xyz: xyz coordinates
    patch: [p1, p2, p3,
            p4, p5, p6,
            p7, p8, p9]
    surface_normal = [(p9-p1) x (p3-p7)] + [(p6-p4) - (p8-p2)]
    return: normal [h, w, 3, b]
    """
    b, h, w, c = xyz.shape
    half_patch = patch_size // 2
    xyz_pad = torch.zeros((b, h + patch_size - 1, w + patch_size - 1, c), dtype=xyz.dtype, device=xyz.device)
    xyz_pad[:, half_patch:-half_patch, half_patch:-half_patch, :] = xyz

    # xyz_left_top = xyz_pad[:, :h, :w, :]  # p1
    # xyz_right_bottom = xyz_pad[:, -h:, -w:, :]# p9
    # xyz_left_bottom = xyz_pad[:, -h:, :w, :]   # p7
    # xyz_right_top = xyz_pad[:, :h, -w:, :]  # p3
    # xyz_cross1 = xyz_left_top - xyz_right_bottom  # p1p9
    # xyz_cross2 = xyz_left_bottom - xyz_right_top  # p7p3

    xyz_left = xyz_pad[:, half_patch:half_patch + h, :w, :]  # p4
    xyz_right = xyz_pad[:, half_patch:half_patch + h, -w:, :]  # p6
    xyz_top = xyz_pad[:, :h, half_patch:half_patch + w, :]  # p2
    xyz_bottom = xyz_pad[:, -h:, half_patch:half_patch + w, :]  # p8
    xyz_horizon = xyz_left - xyz_right  # p4p6
    xyz_vertical = xyz_top - xyz_bottom  # p2p8

    xyz_left_in = xyz_pad[:, half_patch:half_patch + h, 1:w+1, :]  # p4
    xyz_right_in = xyz_pad[:, half_patch:half_patch + h, patch_size-1:patch_size-1+w, :]  # p6
    xyz_top_in = xyz_pad[:, 1:h+1, half_patch:half_patch + w, :]  # p2
    xyz_bottom_in = xyz_pad[:, patch_size-1:patch_size-1+h, half_patch:half_patch + w, :]  # p8
    xyz_horizon_in = xyz_left_in - xyz_right_in  # p4p6
    xyz_vertical_in = xyz_top_in - xyz_bottom_in  # p2p8

    n_img_1 = torch.cross(xyz_horizon_in, xyz_vertical_in, dim=3)
    n_img_2 = torch.cross(xyz_horizon, xyz_vertical, dim=3)

    # re-orient normals consistently
    orient_mask = torch.sum(n_img_1 * xyz, dim=3) > 0
    n_img_1[orient_mask] *= -1
    orient_mask = torch.sum(n_img_2 * xyz, dim=3) > 0
    n_img_2[orient_mask] *= -1

    n_img1_L2 = torch.sqrt(torch.sum(n_img_1 ** 2, dim=3, keepdim=True))
    n_img1_norm = n_img_1 / (n_img1_L2 + 1e-8)

    n_img2_L2 = torch.sqrt(torch.sum(n_img_2 ** 2, dim=3, keepdim=True))
    n_img2_norm = n_img_2 / (n_img2_L2 + 1e-8)

    # average 2 norms
    n_img_aver = n_img1_norm + n_img2_norm
    n_img_aver_L2 = torch.sqrt(torch.sum(n_img_aver ** 2, dim=3, keepdim=True))
    n_img_aver_norm = n_img_aver / (n_img_aver_L2 + 1e-8)
    # re-orient normals consistently
    orient_mask = torch.sum(n_img_aver_norm * xyz, dim=3) > 0
    n_img_aver_norm[orient_mask] *= -1
    n_img_aver_norm_out = n_img_aver_norm.permute((1, 2, 3, 0))  # [h, w, c, b]

    # a = torch.sum(n_img1_norm_out*n_img2_norm_out, dim=2).cpu().numpy().squeeze()
    # plt.imshow(np.abs(a), cmap='rainbow')
    # plt.show()
    return n_img_aver_norm_out#n_img1_norm.permute((1, 2, 3, 0))

def surface_normal_from_depth(depth, intrinsic, valid_mask=None):
    # para depth: depth map, [b, c, h, w]
    b, c, h, w = depth.shape
    # focal_length = focal_length[:, None, None, None]
    depth_filter = torch.nn.functional.avg_pool2d(depth, kernel_size=3, stride=1, padding=1)
    depth_filter = torch.nn.functional.avg_pool2d(depth_filter, kernel_size=3, stride=1, padding=1)
    xyz = depth_to_xyz(depth_filter, intrinsic)
    sn_batch = []
    for i in range(b):
        xyz_i = xyz[i, :][None, :, :, :]
        normal = get_surface_normalv2(xyz_i)
        sn_batch.append(normal)
    sn_batch = torch.cat(sn_batch, dim=3).permute((3, 2, 0, 1))  # [b, c, h, w]
    if valid_mask is not None:
        mask_invalid = (~valid_mask).repeat(1, 3, 1, 1)
        sn_batch[mask_invalid] = 0.0

    return sn_batch

def img_warping(ref_pose, src_pose, virtual_pose_ref_depth, virtual_intrinsic, src_img):
    ref_depth = virtual_pose_ref_depth
    ref_pose = ref_pose
    src_pose = src_pose
    intrinsic = virtual_intrinsic

    mask = ref_depth > 0

    ht, wd = ref_depth.shape
    fx, fy, cx, cy = intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2]

    y, x = torch.meshgrid(torch.arange(ht).float(), torch.arange(wd).float())
    y = y.to(ref_depth.device)
    x = x.to(ref_depth.device)

    i = torch.ones_like(ref_depth).to(ref_depth.device)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    pts_in_norm = torch.stack([X, Y, i], dim=-1)
    pts_in_3D = pts_in_norm * ref_depth.unsqueeze(-1).repeat(1, 1, 3)

    rel_pose = src_pose.inverse() @ ref_pose

    pts_in_3D_tgt = rel_pose[:3, :3] @ pts_in_3D.view(-1, 3).permute(1, 0) + rel_pose[:3, 3].unsqueeze(-1).repeat(1, ht*wd)
    pts_in_norm_tgt = pts_in_3D_tgt / pts_in_3D_tgt[2:, :]

    pts_in_tgt = intrinsic @ pts_in_norm_tgt
    pts_in_tgt = pts_in_tgt.permute(1, 0).view(ht, wd, 3)[:, :, :2]

    pts_in_tgt[:, :, 0] = (pts_in_tgt[:, :, 0] / wd - 0.5) * 2
    pts_in_tgt[:, :, 1] = (pts_in_tgt[:, :, 1] / ht - 0.5) * 2
    warped_ref_img = torch.nn.functional.grid_sample(src_img.unsqueeze(0), pts_in_tgt.unsqueeze(0), mode='nearest', padding_mode="zeros")

    return warped_ref_img

def get_proj_matrix(K, image_size, znear=.01, zfar=1000.):
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    width, height = image_size
    m = np.zeros((4, 4))
    m[0][0] = 2.0 * fx / width
    m[0][1] = 0.0
    m[0][2] = 0.0
    m[0][3] = 0.0

    m[1][0] = 0.0
    m[1][1] = 2.0 * fy / height
    m[1][2] = 0.0
    m[1][3] = 0.0

    m[2][0] = 1.0 - 2.0 * cx / width
    m[2][1] = 2.0 * cy / height - 1.0
    m[2][2] = (zfar + znear) / (znear - zfar)
    m[2][3] = -1.0

    m[3][0] = 0.0
    m[3][1] = 0.0
    m[3][2] = 2.0 * zfar * znear / (znear - zfar)
    m[3][3] = 0.0

    return m.T


def bilinear_sampler(img, coords, mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = torch.nn.functional.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


# def sparse_depth_from_projection(gaussians, viewpoint_cam):
#     pc = gaussians.get_xyz.contiguous()
#     K = viewpoint_cam.K
#     img_height = viewpoint_cam.image_height
#     img_width = viewpoint_cam.image_width
#     znear = 0.1
#     zfar = 1000
#     proj_matrix = get_proj_matrix(K, (img_width, img_height), znear, zfar)
#     proj_matrix = torch.tensor(proj_matrix).cuda().to(torch.float32)
#     w2c = viewpoint_cam.world_view_transform.transpose(0, 1)
#     c2w = w2c.inverse()
#     c2w = c2w @ torch.tensor(np.diag([1., -1., -1., 1.]).astype(np.float32)).cuda()
#     w2c = c2w.inverse()
#     total_m = proj_matrix @ w2c
#     index_buffer, _ = pcpr.forward(pc, total_m.unsqueeze(0), img_width, img_height, 512)
#     sh = index_buffer.shape
#     ind = index_buffer.view(-1).long().cuda()

#     xyz = pc.unsqueeze(0).permute(2,0,1)
#     xyz = xyz.view(xyz.shape[0],-1)
#     proj_xyz_world = torch.index_select(xyz, 1, ind)
#     Rot, Trans = w2c[:3, :3], w2c[:3, 3][..., None]

#     proj_xyz_cam = Rot @ proj_xyz_world + Trans
#     proj_depth = proj_xyz_cam[2,:][None,]
#     proj_depth = proj_depth.view(proj_depth.shape[0], sh[0], sh[1], sh[2]) #[1, 4, 256, 256]
#     proj_depth = proj_depth.permute(1, 0, 2, 3)
#     proj_depth *= -1

#     ##mask获取
#     mask = ind.clone()
#     mask[mask>0] = 1
#     mask = mask.view(1, sh[0], sh[1], sh[2])
#     mask = mask.permute(1,0,2,3)

#     proj_depth = proj_depth * mask

#     return proj_depth.squeeze()

# project the reference point cloud into the source view, then project back
#extrinsics here refers c2w
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    batch, height, width = depth_ref.shape
    
    ## step1. project reference pixels to the source view
    # reference view x, y
    y_ref, x_ref = torch.meshgrid(torch.arange(0, height).to(depth_ref.device), torch.arange(0, width).to(depth_ref.device))
    x_ref = x_ref.unsqueeze(0).repeat(batch,  1, 1)
    y_ref = y_ref.unsqueeze(0).repeat(batch,  1, 1)
    x_ref, y_ref = x_ref.reshape(batch, -1), y_ref.reshape(batch, -1)
    # reference 3D space

    A = torch.inverse(intrinsics_ref)
    B = torch.stack((x_ref, y_ref, torch.ones_like(x_ref).to(x_ref.device)), dim=1) * depth_ref.reshape(batch, 1, -1)
    xyz_ref = torch.matmul(A, B)

    # source 3D space
    xyz_src = torch.matmul(torch.matmul(torch.inverse(extrinsics_src), extrinsics_ref),
                        torch.cat((xyz_ref, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1))[:, :3]
    # source view x, y
    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:, :2] / K_xyz_src[:, 2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[:, 0].reshape([batch, height, width]).float()
    y_src = xy_src[:, 1].reshape([batch, height, width]).float()

    # print(x_src, y_src)
    sampled_depth_src = bilinear_sampler(depth_src.view(batch, 1, height, width), torch.stack((x_src, y_src), dim=-1).view(batch, height, width, 2))

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = torch.matmul(torch.inverse(intrinsics_src),
                        torch.cat((xy_src, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1) * sampled_depth_src.reshape(batch, 1, -1))
    # reference 3D space
    xyz_reprojected = torch.matmul(torch.matmul(torch.inverse(extrinsics_ref), extrinsics_src),
                                torch.cat((xyz_src, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1))[:, :3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[:, 2].reshape([batch, height, width]).float()
    K_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:, :2] / K_xyz_reprojected[:, 2:3]
    x_reprojected = xy_reprojected[:, 0].reshape([batch, height, width]).float()
    y_reprojected = xy_reprojected[:, 1].reshape([batch, height, width]).float()

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src, thre1=1, thre2=0.01):
    batch, height, width = depth_ref.shape
    y_ref, x_ref = torch.meshgrid(torch.arange(0, height).to(depth_ref.device), torch.arange(0, width).to(depth_ref.device))
    x_ref = x_ref.unsqueeze(0).repeat(batch,  1, 1)
    y_ref = y_ref.unsqueeze(0).repeat(batch,  1, 1)
    inputs = [depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src]
    outputs = reproject_with_depth(*inputs)
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = outputs
    # check |p_reproj-p_1| < 1
    dist = torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = torch.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = torch.logical_and(dist < thre1, relative_depth_diff < thre2)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff

def depth_propagation(viewpoint_cam, rendered_depth, viewpoint_stack, src_idxs, dataset, patch_size):
    
    depth_min = 0.1
    if dataset == 'waymo':
        depth_max = 80
    elif dataset == '360':
        depth_max = 20
    else:
        depth_max = 20

    images = list()
    intrinsics = list()
    poses = list()
    depth_intervals = list()
    
    images.append((viewpoint_cam.original_image * 255).permute((1, 2, 0)).to(torch.uint8))
    intrinsics.append(viewpoint_cam.K)
    poses.append(viewpoint_cam.world_view_transform.transpose(0, 1))
    depth_interval = torch.tensor([depth_min, (depth_max-depth_min)/192.0, 192.0, depth_max])
    depth_intervals.append(depth_interval)
    
    depth = rendered_depth.unsqueeze(-1)
    normal = torch.zeros_like(depth)
    
    for idx, src_idx in enumerate(src_idxs):
        src_viewpoint = viewpoint_stack[src_idx]
        images.append((src_viewpoint.original_image * 255).permute((1, 2, 0)).to(torch.uint8))
        intrinsics.append(src_viewpoint.K)
        poses.append(src_viewpoint.world_view_transform.transpose(0, 1))
        depth_intervals.append(depth_interval)
        
    images = torch.stack(images)
    intrinsics = torch.stack(intrinsics)
    poses = torch.stack(poses)
    depth_intervals = torch.stack(depth_intervals)

    results = propagate(images, intrinsics, poses, depth, normal, depth_intervals, patch_size)
    propagated_depth = results[0].to(rendered_depth.device)
    propagated_normal = results[1:4].to(rendered_depth.device).permute(1, 2, 0)
    
    return propagated_depth, propagated_normal

    
def generate_edge_mask(propagated_depth, patch_size):
    # img gradient
    x_conv = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).float().cuda()
    y_conv = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3).float().cuda()
    gradient_x = torch.abs(torch.nn.functional.conv2d(propagated_depth.unsqueeze(0).unsqueeze(0), x_conv, padding=1))
    gradient_y = torch.abs(torch.nn.functional.conv2d(propagated_depth.unsqueeze(0).unsqueeze(0), y_conv, padding=1))
    gradient = gradient_x + gradient_y

    # edge mask
    edge_mask = (gradient > 5).float()

    # dilation
    kernel = torch.ones(1, 1, patch_size, patch_size).float().cuda()
    dilated_mask = torch.nn.functional.conv2d(edge_mask, kernel, padding=(patch_size-1)//2)
    dilated_mask = torch.round(dilated_mask).squeeze().to(torch.bool)
    dilated_mask = ~dilated_mask

    return dilated_mask