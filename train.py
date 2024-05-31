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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import vis_depth, read_propagted_depth
from gaussian_renderer import render, network_gui
from utils.graphics_utils import depth_propagation, check_geometric_consistency
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, load_pairs_relation
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import imageio
import numpy as np
import torchvision
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    
    #read the overlapping txt
    if opt.dataset == '360' and opt.depth_loss:
        pairs = load_pairs_relation(opt.pair_path)
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    if not opt.ordered:
        distances = torch.zeros(len(viewpoint_stack), len(viewpoint_stack), device="cuda")
        for i, view in tqdm(enumerate(viewpoint_stack), desc="Calculating distances"):
            for j, other_view in enumerate(viewpoint_stack):
                distances[j, i] = distances[i, j] = torch.norm(view.camera_center - other_view.camera_center)
        neighbor_indices = torch.argsort(distances, dim=1)[:, :5]
    else:
        neighbor_indices = None

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # depth_loss_fn = ScaleAndShiftInvariantLoss(alpha=0.1, scales=1)
    propagated_iteration_begin = opt.propagated_iteration_begin
    propagated_iteration_after = opt.propagated_iteration_after
    after_propagated = False
    propagation_dict = {}
    for i in range(0, len(viewpoint_stack), 1):
        propagation_dict[viewpoint_stack[i].image_name] = False

    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        randidx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack[randidx]
        
        if opt.depth_loss:
            if opt.dataset == '360':
                src_idxs = pairs[randidx]
            elif opt.ordered:
                # intervals = [-6, -3, 3, 6]
                if opt.dataset == 'waymo':
                    intervals = [-2, -1, 1, 2]
                elif opt.dataset == 'scannet':
                    intervals = [-10, -5, 5, 10]
                elif opt.dataset == 'free':
                    intervals = [-2, -1, 1, 2]
                src_idxs = [randidx+itv for itv in intervals if ((itv + randidx > 0) and (itv + randidx < len(viewpoint_stack)))]
            else:
                src_idxs = neighbor_indices[randidx].tolist()

        #propagate the gaussians first
        with torch.no_grad():
           if opt.depth_loss and iteration > propagated_iteration_begin and iteration < propagated_iteration_after and (iteration % opt.propagation_interval == 0 and not propagation_dict[viewpoint_cam.image_name]):
            # if opt.depth_loss and iteration > propagated_iteration_begin and iteration < propagated_iteration_after and (iteration % opt.propagation_interval == 0):
                propagation_dict[viewpoint_cam.image_name] = True

                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, 
                            return_normal=opt.normal_loss, return_opacity=False, return_depth=opt.depth_loss or opt.depth2normal_loss)

                projected_depth = render_pkg["render_depth"]

                # get the opacity that less than the threshold, propagate depth in these region
                if viewpoint_cam.sky_mask is not None:
                    sky_mask = viewpoint_cam.sky_mask.to(opacity_mask.device).to(torch.bool)
                else:
                    sky_mask = None

                # get the propagated depth
                propagated_depth, normal = depth_propagation(viewpoint_cam, projected_depth, viewpoint_stack, src_idxs, opt.dataset, opt.patch_size)

                # cache the propagated_depth
                viewpoint_cam.depth = propagated_depth

                #transform normal to camera coordinate
                R_w2c = torch.tensor(viewpoint_cam.R.T, dtype=torch.float, device="cuda")
                # R_w2c[:, 1:] *= -1
                normal = (R_w2c @ normal.view(-1, 3).permute(1, 0)).view(3, viewpoint_cam.image_height, viewpoint_cam.image_width)                
                valid_mask = propagated_depth != 300

                # calculate the abs rel depth error of the propagated depth and rendered depth & render color error
                render_depth = render_pkg['render_depth']
                abs_rel_error = torch.abs(propagated_depth - render_depth) / propagated_depth
                abs_rel_error_threshold = opt.depth_error_max_threshold - (opt.depth_error_max_threshold - opt.depth_error_min_threshold) * (iteration - propagated_iteration_begin) / (propagated_iteration_after - propagated_iteration_begin)
                # color error
                # render_color = render_pkg['render']

                # color_error = torch.abs(render_color - viewpoint_cam.original_image)
                # color_error = color_error.mean(dim=0).squeeze()
                error_mask = (abs_rel_error > abs_rel_error_threshold)

                # # calculate the photometric consistency
                ref_K = viewpoint_cam.K
                #c2w
                ref_pose = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()
                
                # calculate the geometric consistency
                geometric_counts = None
                for idx, src_idx in enumerate(src_idxs):
                    src_viewpoint = viewpoint_stack[src_idx]
                    #c2w
                    src_pose = src_viewpoint.world_view_transform.transpose(0, 1).inverse()
                    src_K = src_viewpoint.K

                    if src_viewpoint.depth is None:
                        src_render_pkg = render(src_viewpoint, gaussians, pipe, bg, 
                                return_normal=opt.normal_loss, return_opacity=False, return_depth=opt.depth_loss or opt.depth2normal_loss)
                        src_projected_depth = src_render_pkg['render_depth']
                    
                    #get the src_depth first
                        src_depth, src_normal = depth_propagation(src_viewpoint, src_projected_depth, viewpoint_stack, src_idxs, opt.dataset, opt.patch_size)
                        src_viewpoint.depth = src_depth
                    else:
                        src_depth = src_viewpoint.depth
                        
                    mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff = check_geometric_consistency(propagated_depth.unsqueeze(0), ref_K.unsqueeze(0), 
                                                                                                                 ref_pose.unsqueeze(0), src_depth.unsqueeze(0), 
                                                                                                                 src_K.unsqueeze(0), src_pose.unsqueeze(0), thre1=2, thre2=0.01)
                    
                    if geometric_counts is None:
                        geometric_counts = mask.to(torch.uint8)
                    else:
                        geometric_counts += mask.to(torch.uint8)
                        
                cost = geometric_counts.squeeze()
                cost_mask = cost >= 2       
                
                normal[~(cost_mask.unsqueeze(0).repeat(3, 1, 1))] = -10
                viewpoint_cam.normal = normal
                
                propagated_mask = valid_mask & error_mask & cost_mask
                if sky_mask is not None:
                    propagated_mask = propagated_mask & sky_mask

                propagated_depth[~cost_mask] = 300 
                # propagated_mask = propagated_mask & edge_mask
                propagated_depth[~propagated_mask] = 300
  
                if propagated_mask.sum() > 100:
                    gaussians.densify_from_depth_propagation(viewpoint_cam, propagated_depth, propagated_mask.to(torch.bool), gt_image) 
                
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        #render_pkg = render(viewpoint_cam, gaussians, pipe, bg, return_normal=args.normal_loss)
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, 
                            return_normal=opt.normal_loss, return_opacity=True, return_depth=opt.depth_loss or opt.depth2normal_loss)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # opacity mask
        if iteration < opt.propagated_iteration_begin and opt.depth_loss:
            opacity_mask = render_pkg['render_opacity'] > 0.999
            opacity_mask = opacity_mask.unsqueeze(0).repeat(3, 1, 1)
        else:
            opacity_mask = render_pkg['render_opacity'] > 0.0
            opacity_mask = opacity_mask.unsqueeze(0).repeat(3, 1, 1)

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image[opacity_mask], gt_image[opacity_mask])
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image, mask=opacity_mask))

        # flatten loss
        if opt.flatten_loss:
            scales = gaussians.get_scaling
            min_scale, _ = torch.min(scales, dim=1)
            min_scale = torch.clamp(min_scale, 0, 30)
            flatten_loss = torch.abs(min_scale).mean()
            loss += opt.lambda_flatten * flatten_loss

        # opacity loss
        if opt.sparse_loss:
            opacity = gaussians.get_opacity
            opacity = opacity.clamp(1e-6, 1-1e-6)
            log_opacity = opacity * torch.log(opacity)
            log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
            sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[visibility_filter].mean()
            loss += opt.lambda_sparse * sparse_loss

        if opt.normal_loss:
            rendered_normal = render_pkg['render_normal']
            if viewpoint_cam.normal is not None:
                normal_gt = viewpoint_cam.normal.cuda()
                if viewpoint_cam.sky_mask is not None:
                    filter_mask = viewpoint_cam.sky_mask.to(normal_gt.device).to(torch.bool)
                    normal_gt[~(filter_mask.unsqueeze(0).repeat(3, 1, 1))] = -10
                filter_mask = (normal_gt != -10)[0, :, :].to(torch.bool)

                l1_normal = torch.abs(rendered_normal - normal_gt).sum(dim=0)[filter_mask].mean()
                cos_normal = (1. - torch.sum(rendered_normal * normal_gt, dim = 0))[filter_mask].mean()
                loss += opt.lambda_l1_normal * l1_normal + opt.lambda_cos_normal * cos_normal

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if not torch.isnan(loss):
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 2000, 7000, 20000, 50000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 7000, 20000, 50000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
