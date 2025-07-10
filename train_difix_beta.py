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
import json
import time
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import numpy as np
from PIL import Image
import imageio
import sys
from scipy.spatial.transform import Rotation, Slerp

from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim
from scene import Scene, BetaModel
from scene.cameras import Camera
from utils.general_utils import safe_state
from arguments import ModelParams, ViewerParams, OptimizationParams
from scene.beta_model import build_scaling_rotation
import viser
from scene.beta_viewer import BetaViewer
from Difix3D.src.pipeline_difix import DifixPipeline

# Helper functions and classes from Difix3D examples
# ===============================================================================================

def normalize(v):
    return v / np.linalg.norm(v)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def generate_interpolated_path(camtoworlds, n_interp):
    """Generates an interpolated path between a set of camera poses."""
    key_rots = Rotation.from_matrix(camtoworlds[:, :3, :3])
    key_times = np.linspace(0, 1, len(camtoworlds))
    slerp = Slerp(key_times, key_rots)
    interp_times = np.linspace(0, 1, n_interp)
    interp_rots = slerp(interp_times).as_matrix()
    interp_trans = np.array([np.interp(interp_times, key_times, camtoworlds[:, i, 3]) for i in range(3)]).T
    
    ready_poses = np.zeros((n_interp, 4, 4))
    ready_poses[:, :3, :3] = interp_rots
    ready_poses[:, :3, 3] = interp_trans
    ready_poses[:, 3, 3] = 1.0
    return ready_poses

class CameraPoseInterpolator:
    """Helper class to interpolate and manage camera poses."""
    def __init__(self, rotation_weight=1.0, translation_weight=1.0):
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight

    def find_nearest_assignments(self, poses1, poses2):
        """Find nearest pose in poses1 for each pose in poses2."""
        poses1_trans = poses1[:, :3, 3]
        poses2_trans = poses2[:, :3, 3]
        
        dist_matrix = np.linalg.norm(poses1_trans[:, np.newaxis, :] - poses2_trans[np.newaxis, :, :], axis=2)
        return np.argmin(dist_matrix, axis=0)

    def shift_poses(self, novel_poses, ref_poses, distance=0.1):
        """Slightly shift novel poses towards their nearest reference poses."""
        assignments = self.find_nearest_assignments(ref_poses, novel_poses)
        shifted_poses = np.copy(novel_poses)
        
        for i in range(len(novel_poses)):
            ref_pose = ref_poses[assignments[i]]
            
            # Interpolate translation
            shifted_poses[i, :3, 3] = (1 - distance) * novel_poses[i, :3, 3] + distance * ref_pose[:3, 3]
            
            # Interpolate rotation via Slerp
            key_rots = Rotation.from_matrix([novel_poses[i, :3, :3], ref_pose[:3, :3]])
            key_times = [0, 1]
            slerp = Slerp(key_times, key_rots)
            shifted_poses[i, :3, :3] = slerp([distance]).as_matrix()[0]
            
        return shifted_poses

# ===============================================================================================

def render_traj(args, model: BetaModel, scene: Scene, poses, tag="novel"):
    """Renders a trajectory of images from the given camera poses."""
    print(f"[{'RENDER'}] Rendering trajectory '{tag}'...")
    render_dir = os.path.join(args.model_path, "renders", tag)
    
    # Ensure all directories exist
    os.makedirs(os.path.join(render_dir, "Pred"), exist_ok=True)
    os.makedirs(os.path.join(render_dir, "Alpha"), exist_ok=True)
    
    # Use the first camera as a template for intrinsics
    cam_template = scene.getTrainCameras()[0]
    
    for i, pose in enumerate(tqdm(poses, desc=f"Rendering {tag} trajectory")):
        # Create a new Camera object for the novel pose
        uid = -1 # Novel views don't have a real UID
        novel_camera = Camera(
            colmap_id=uid, R=pose[:3, :3], T=pose[:3, 3], 
            FoVx=cam_template.FoVx, FoVy=cam_template.FoVy,
            image=torch.empty(3, cam_template.image_height, cam_template.image_width), # Placeholder image
            gt_alpha_mask=None,
            image_name=f"novel_{tag}_{i}",
            image_path="", # Add placeholder for image_path
            uid=uid,
        )
        
        with torch.no_grad():
            render_pkg = model.render(novel_camera)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            alpha = render_pkg["alpha"]

        # Save rendered prediction
        pred_path = os.path.join(render_dir, "Pred", f"{i:04d}.png")
        image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(pred_path, image_np)

        # Save alpha mask
        alpha_path = os.path.join(render_dir, "Alpha", f"{i:04d}.png")
        alpha_np = (alpha.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(alpha_np, mode='L').save(alpha_path)

def fix(args, iteration, model: BetaModel, scene: Scene, difix_pipe, interpolator, novel_data):
    """Renders, cleans, and creates a new dataset from fixed images."""
    print(f"\n[{'FIX'}] Iteration {iteration}: Running artifact fixing step...")

    # 1. Generate new camera poses by interpolating between existing training views
    print(f"[{'FIX'}] Generating novel camera poses...")
    train_poses = np.stack([c.world_view_transform.T.cpu() for c in scene.getTrainCameras()])
    if not novel_data: # First fix step, create a truly novel path
        current_novel_poses = generate_interpolated_path(train_poses, len(train_poses))
    else: # Subsequent steps, refine the last set of poses
        current_novel_poses = novel_data[-1]['poses']

    # Shift poses slightly towards the original data for stability
    shifted_poses = interpolator.shift_poses(current_novel_poses, train_poses, distance=0.1)
    
    # 2. Render the trajectory for these new poses
    render_tag = f"novel_{iteration}"
    render_traj(args, model, scene, shifted_poses, tag=render_tag)
    
    # 3. Find nearest reference images and run Difix to clean the rendered images
    print(f"[{'FIX'}] Cleaning rendered images with Difix...")
    render_dir = os.path.join(args.model_path, "renders", render_tag)
    image_paths = [os.path.join(render_dir, "Pred", f"{i:04d}.png") for i in range(len(shifted_poses))]
    
    ref_indices = interpolator.find_nearest_assignments(train_poses, shifted_poses)
    ref_image_paths = [scene.getTrainCameras()[i].image_path for i in ref_indices]
    
    fixed_dir = os.path.join(render_dir, "Fixed")
    os.makedirs(fixed_dir, exist_ok=True)

    for i in tqdm(range(len(shifted_poses)), desc="Fixing artifacts"):
        image = Image.open(image_paths[i]).convert("RGB")
        ref_image = Image.open(ref_image_paths[i]).convert("RGB")
        
        with torch.no_grad():
            output_image = difix_pipe(
                prompt=args.difix_prompt, 
                image=image, 
                ref_image=ref_image, 
                num_inference_steps=1,
                timesteps=[199],
                guidance_scale=0.2
            ).images[0]
            
        output_image = output_image.resize(image.size, Image.LANCZOS)
        output_image.save(os.path.join(fixed_dir, f"{i:04d}.png"))
    
    # 4. Create a new dataset of Camera objects from the fixed images
    print(f"[{'FIX'}] Creating new dataset from {len(shifted_poses)} fixed images...")
    fixed_cameras = []
    cam_template = scene.getTrainCameras()[0]
    
    for i, pose in enumerate(shifted_poses):
        uid = -1 # Novel views don't have a real UID
        image_path = os.path.join(fixed_dir, f"{i:04d}.png")
        # The Camera class expects an image tensor, so we load it
        img = np.array(Image.open(image_path)) / 255.0
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)

        fixed_camera = Camera(
            colmap_id=uid, R=pose[:3, :3], T=pose[:3, 3],
            FoVx=cam_template.FoVx, FoVy=cam_template.FoVy,
            image=img_tensor,
            gt_alpha_mask=None,
            image_name=f"fixed_{iteration}_{i}",
            image_path=image_path, # Add placeholder for image_path
            uid=uid,
        )
        fixed_cameras.append(fixed_camera)
        
    novel_data.append({'cameras': fixed_cameras, 'poses': shifted_poses})
    print(f"[{'FIX'}] Fix step complete. Added new dataset.")


def training(args):
    first_iter = 0
    prepare_output_and_logger(args)
    beta_model = BetaModel(args.sh_degree, args.sb_number)
    scene = Scene(args, beta_model)
    beta_model.training_setup(args)

    if args.start_checkpoint:
        (model_params, first_iter) = torch.load(args.start_checkpoint)
        beta_model.restore(model_params, args)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not args.disable_viewer:
        server = viser.ViserServer(port=args.port, verbose=False)
        viewer = BetaViewer(
            server=server,
            render_fn=lambda camera_state, render_tab_state: beta_model.view(
                camera_state, render_tab_state, args.center
            ),
            mode="training",
            share_url=args.share_url,
        )

    # Initialize Difix pipeline and helpers if fix_steps are scheduled
    difix_pipe, interpolator, novel_data = None, None, []
    if args.fix_steps:
        print("[INIT] Initializing Difix pipeline for artifact fixing...")
        difix_pipe = DifixPipeline.from_pretrained(
            "nvidia/difix_ref", trust_remote_code=True, torch_dtype=torch.float16
        ).to("cuda")
        difix_pipe.set_progress_bar_config(disable=True)
        interpolator = CameraPoseInterpolator()
        print("[INIT] Difix pipeline initialized.")


    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, args.iterations), desc="Training progress")
    iteration = first_iter

    while iteration < args.iterations:
        # Run fix step at specified iterations
        if iteration in args.fix_steps:
            fix(args, iteration, beta_model, scene, difix_pipe, interpolator, novel_data)
            # After fixing, we might want to clear the old viewpoint stack to include new views
            viewpoint_stack = None

        iter_start.record()
        if not args.disable_viewer:
            while viewer.state == "paused":
                time.sleep(0.01)
            viewer.lock.acquire()
            tic = time.time()

        xyz_lr = beta_model.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            beta_model.oneupSHdegree()

        # Decide whether to train on original data or fixed "novel" data
        use_novel_data = len(novel_data) > 0 and randint(0, 2) == 0 # 33% chance to use novel data
        
        if use_novel_data:
            # Pick a random camera from the latest set of fixed images
            fixed_dataset = novel_data[-1]['cameras']
            viewpoint_cam = fixed_dataset[randint(0, len(fixed_dataset) - 1)]
            gt_image = viewpoint_cam.original_image # This is the "fixed" image
        else:
            # Pick a random Camera from the original training set
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            gt_image = viewpoint_cam.original_image.cuda()

        beta_model.background = (
            torch.rand((3), device="cuda") if args.random_background else background
        )
        render_pkg = beta_model.render(viewpoint_cam)
        image = render_pkg["render"]

        # Calculate loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (
            1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        )
        
        # Apply a different weight if we are training on the "fixed" data
        if use_novel_data:
            loss *= args.lambda_difix

        if args.densify_from_iter < iteration < args.densify_until_iter:
            loss += args.opacity_reg * torch.abs(beta_model.get_opacity).mean()
            loss += args.scale_reg * torch.abs(beta_model.get_scaling).mean()

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            postfix = {
                "Loss": f"{ema_loss_for_log:.7f}",
                "Beta": f"{beta_model._beta.mean().item():.2f}",
                "Source": "Fixed" if use_novel_data else "Original"
            }
            progress_bar.set_postfix(postfix)
            progress_bar.update(1)
            iteration += 1

            if iteration in args.save_iterations:
                print(f"\n[ITER {iteration}] Saving beta_model")
                scene.save(iteration)

            if (
                iteration < args.densify_until_iter
                and iteration > args.densify_from_iter
                and iteration % args.densification_interval == 0
            ):
                dead_mask = (beta_model.get_opacity <= 0.005).squeeze(-1)
                beta_model.relocate_gs(dead_mask=dead_mask)
                beta_model.add_new_gs(cap_max=args.cap_max)

                L = build_scaling_rotation(
                    beta_model.get_scaling, beta_model.get_rotation
                )
                actual_covariance = L @ L.transpose(1, 2)
                noise = (
                    torch.randn_like(beta_model._xyz)
                    * (torch.pow(1 - beta_model.get_opacity, 100))
                    * args.noise_lr
                    * xyz_lr
                )
                noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                beta_model._xyz.add_(noise)

            beta_model.optimizer.step()
            beta_model.optimizer.zero_grad(set_to_none=True)

            if not args.disable_viewer:
                num_train_rays_per_step = gt_image.numel()
                viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic + 1e-8)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_sec
                viewer.update(iteration, num_train_rays_per_step)

    progress_bar.close()
    print("\nTraining complete.\n")

    # Evaluation and compression logic remains the same
    if args.eval:
        print("\nEvaluating Last Checkpoint Performance\n")
        last_iter = args.save_iterations[-1]
        beta_model.load_ply(
            os.path.join(scene.model_path, f"point_cloud/iteration_{last_iter}/point_cloud.ply")
        )
        result = scene.eval()
        with open(
            os.path.join(scene.model_path, f"point_cloud/iteration_{last_iter}/metrics.json"),
            "w",
        ) as f:
            json.dump(result, f, indent=True)

    if args.compress:
        last_iter = args.save_iterations[-1]
        print(f"Compressing model at iteration {last_iter}...")
        beta_model.save_png(
            os.path.join(scene.model_path, f"point_cloud/iteration_{last_iter}")
        )

    if not args.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)

def prepare_output_and_logger(args):
    if not args.model_path:
        args.model_path = os.path.join("./output/", os.path.basename(args.source_path))
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    ModelParams(parser), OptimizationParams(parser), ViewerParams(parser)
    
    # Add arguments for Difix artifact cleaning
    fix_group = parser.add_argument_group("Fixing Parameters")
    fix_group.add_argument("--fix_steps", nargs="+", type=int, default=[], help="List of iterations to run the artifact fixing step.")
    fix_group.add_argument("--difix_prompt", type=str, default="remove degradation", help="Prompt for the Difix model.")
    fix_group.add_argument("--lambda_difix", type=float, default=0.2, help="Loss weight for the fixed 'novel' data.")
    
    # Original arguments from train.py
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--no-compress", action="store_false", dest="compress", help="Disable compression")
    parser.set_defaults(compress=True)
    parser.add_argument("--share_url", action="store_true", help="Share URL for the viewer")
    parser.add_argument("--center", action="store_true", help="Center the model in the viewer")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    training(args) 