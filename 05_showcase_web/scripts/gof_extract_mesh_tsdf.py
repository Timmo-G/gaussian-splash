import torch
from scene import Scene
import os
from os import makedirs
from gaussian_renderer import render
import random
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import open3d as o3d
import open3d.core as o3c
import math
        
def tsdf_fusion(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    kernel_size,
    o3d_device_name,
    voxel_size,
    block_resolution,
    block_count,
    alpha_thres,
    depth_max,
    view_stride,
    max_views,
):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "tsdf")

    makedirs(render_path, exist_ok=True)
    o3d_device = o3d.core.Device(o3d_device_name)
    selected_views = views[::max(1, view_stride)]
    if max_views > 0:
        selected_views = selected_views[:max_views]
    print(
        f"TSDF fusion: {len(selected_views)}/{len(views)} views, device={o3d_device}, "
        f"voxel_size={voxel_size}, block_resolution={block_resolution}, block_count={block_count}"
    )
    
    vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=voxel_size,
            block_resolution=block_resolution,
            block_count=block_count,
            device=o3d_device)
    
    with torch.no_grad():
        for _, view in enumerate(tqdm(selected_views, desc="Rendering progress")):
            
            rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size)["render"]
            
            depth = rendering[6:7, :, :]
            alpha = rendering[7:8, :, :]
            rgb = rendering[:3, :, :]
            
            if view.gt_alpha_mask is not None:
                depth[(view.gt_alpha_mask < 0.5)] = 0
            
            depth[(alpha < alpha_thres)] = 0
            
            W = view.image_width
            H = view.image_height
            ndc2pix = torch.tensor([
                [W / 2, 0, 0, (W-1) / 2],
                [0, H / 2, 0, (H-1) / 2],
                [0, 0, 0, 1]]).float().cuda().T
            intrins =  (view.projection_matrix @ ndc2pix)[:3,:3].T
            intrinsic=o3d.camera.PinholeCameraIntrinsic(
                width=W,
                height=H,
                cx = intrins[0,2].item(),
                cy = intrins[1,2].item(), 
                fx = intrins[0,0].item(), 
                fy = intrins[1,1].item()
            )
            
            extrinsic = np.asarray((view.world_view_transform.T).cpu().numpy())
            
            o3d_color = o3d.t.geometry.Image(np.asarray(rgb.permute(1,2,0).cpu().numpy(), order="C"))
            o3d_depth = o3d.t.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C"))
            o3d_color = o3d_color.to(o3d_device)
            o3d_depth = o3d_depth.to(o3d_device)

            intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix, o3d.core.Dtype.Float64, o3d_device)
            extrinsic = o3d.core.Tensor(extrinsic, o3d.core.Dtype.Float64, o3d_device)
            
            frustum_block_coords = vbg.compute_unique_block_coordinates(
                o3d_depth, intrinsic, extrinsic, 1.0, depth_max)

            vbg.integrate(frustum_block_coords, o3d_depth, o3d_color, intrinsic,
                          intrinsic, extrinsic, 1.0, depth_max)
            
        mesh = vbg.extract_triangle_mesh().to_legacy()
        
        # write mesh
        out_path = f"{render_path}/tsdf.ply"
        o3d.io.write_triangle_mesh(out_path, mesh)
        print(f"Wrote {out_path}")
            
            
def extract_mesh(
    dataset : ModelParams,
    iteration : int,
    pipeline : PipelineParams,
    o3d_device : str,
    voxel_size : float,
    block_resolution : int,
    block_count : int,
    alpha_thres : float,
    depth_max : float,
    view_stride : int,
    max_views : int,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    
        train_cameras = scene.getTrainCameras()
    
        gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size
        
        cams = train_cameras
        tsdf_fusion(
            dataset.model_path,
            "test",
            iteration,
            cams,
            gaussians,
            pipeline,
            background,
            kernel_size,
            o3d_device,
            voxel_size,
            block_resolution,
            block_count,
            alpha_thres,
            depth_max,
            view_stride,
            max_views,
        )

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--o3d_device", default="CPU:0", type=str)
    parser.add_argument("--voxel_size", default=0.01, type=float)
    parser.add_argument("--block_resolution", default=16, type=int)
    parser.add_argument("--block_count", default=50000, type=int)
    parser.add_argument("--alpha_thres", default=0.5, type=float)
    parser.add_argument("--depth_max", default=6.0, type=float)
    parser.add_argument("--view_stride", default=1, type=int)
    parser.add_argument("--max_views", default=0, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    extract_mesh(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.o3d_device,
        args.voxel_size,
        args.block_resolution,
        args.block_count,
        args.alpha_thres,
        args.depth_max,
        args.view_stride,
        args.max_views,
    )
