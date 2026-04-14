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
import os.path as osp
import csv
import torch
from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml

sys.path.append("./")
sys.path.append("./src")
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state
from r2_gaussian.utils.cfg_utils import load_config
from r2_gaussian.utils.log_utils import prepare_output_and_logger
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss
from r2_gaussian.utils.image_utils import metric_vol, metric_proj
from r2_gaussian.utils.plot_utils import show_two_slice


def _psnr_from_mse(mse: float) -> float:
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _projection_eval(cameras, gaussians, pipe):
    if cameras is None or len(cameras) == 0:
        return float("nan"), float("nan")
    proj_l1 = []
    proj_mse = []
    for cam in cameras:
        render_pkg = render(cam, gaussians, pipe)
        pred = render_pkg["render"]
        gt = cam.original_image.cuda()
        proj_l1.append(float(torch.mean(torch.abs(pred - gt)).item()))
        proj_mse.append(float(torch.mean((pred - gt) ** 2).item()))
    loss_val = float(np.mean(proj_l1))
    psnr_val = _psnr_from_mse(float(np.mean(proj_mse)))
    return loss_val, psnr_val


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    eval_interval=100,
    early_stop=False,
    debug_metrics_csv="",
    disable_densify=False,
    disable_prune=False,
):
    first_iter = 0

    # Set up dataset
    scene = Scene(dataset, shuffle=False)

    # Set up some parameters
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world
    queryfunc = lambda x: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
    )

    # Set up Gaussians
    gaussians = GaussianModel(scale_bound)
    initialize_gaussian(gaussians, dataset, None)
    scene.gaussians = gaussians
    gaussians.training_setup(opt)
    if checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Load checkpoint {osp.basename(checkpoint)}.")

    # Set up loss
    use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

    # Train
    protected_organs: list[int] = []
    if getattr(opt, "protected_organs", ""):
        protected_organs = [int(x) for x in str(opt.protected_organs).split(",") if x.strip() != ""]
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(0, opt.iterations), desc="Train", leave=False)
    progress_bar.update(first_iter)
    first_iter += 1
    loss_history: dict[int, float] = {}
    plateau_count = 0
    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()
    fixed_val_cameras = (
        test_cameras[: min(3, len(test_cameras))]
        if test_cameras and len(test_cameras) > 0
        else train_cameras[: min(3, len(train_cameras))]
    )
    debug_csv_path = debug_metrics_csv.strip() if isinstance(debug_metrics_csv, str) else ""
    if not debug_csv_path:
        debug_csv_path = osp.join(scene.model_path, "debug_metrics.csv")
    debug_csv_dir = osp.dirname(debug_csv_path)
    if debug_csv_dir:
        os.makedirs(debug_csv_dir, exist_ok=True)
    with open(debug_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "iter",
                "proj_loss_train",
                "proj_loss_val",
                "proj_psnr_val",
                "global_psnr",
                "voxel_mae",
                "voxel_mean",
                "voxel_std",
                "voxel_min",
                "voxel_max",
                "voxel_p1",
                "voxel_p50",
                "voxel_p99",
                "num_gaussians",
                "density_mean",
                "density_std",
                "scale_mean",
                "scale_std",
                "densify_added",
                "prune_removed",
                "lr_xyz",
            ]
        )
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # Update learning rate
        gaussians.update_learning_rate(iteration)

        # Get one camera for training
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render X-ray projection
        render_pkg = render(viewpoint_cam, gaussians, pipe)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Compute loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss = {"total": 0.0}
        render_loss = l1_loss(image, gt_image)
        loss["render"] = render_loss
        loss["total"] += loss["render"]
        if opt.lambda_dssim > 0:
            loss_dssim = 1.0 - ssim(image, gt_image)
            loss["dssim"] = loss_dssim
            loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim
        # 3D TV loss
        if use_tv:
            # Randomly get the tiny volume center
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                bbox[1] - tv_vol_sVoxel - bbox[0]
            ) * torch.rand(3)
            vol_pred = query(
                gaussians,
                tv_vol_center,
                tv_vol_nVoxel,
                tv_vol_sVoxel,
                pipe,
            )["vol"]
            loss_tv = tv_3d_loss(vol_pred, reduction="mean")
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + opt.lambda_tv * loss_tv

        loss["total"].backward()

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # Adaptive control
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            densify_added = 0
            prune_removed = 0
            if iteration < opt.densify_until_iter:
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    adaptive_stats = gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.density_min_threshold,
                        opt.max_screen_size,
                        max_scale,
                        opt.max_num_gaussians,
                        densify_scale_threshold,
                        bbox,
                        protected_organs=protected_organs if getattr(opt, "organ_aware_prune", False) else None,
                        background_organ=int(getattr(opt, "background_organ", 0)),
                        background_density_scale=float(getattr(opt, "background_density_scale", 2.0)),
                        enable_densify=not bool(disable_densify),
                        enable_prune=not bool(disable_prune),
                    )
                    densify_added = int(adaptive_stats.get("densify_added", 0))
                    prune_removed = int(adaptive_stats.get("prune_removed", 0))
            if gaussians.get_density.shape[0] == 0:
                raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )

            # Optimization
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            proj_loss_now = float(loss["render"].item())
            loss_history[iteration] = proj_loss_now

            # Save gaussians
            if iteration in saving_iterations or iteration == opt.iterations:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, queryfunc)

            # Save checkpoints
            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    ckpt_save_path + "/chkpnt" + str(iteration) + ".pth",
                )

            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss['total'].item():.1e}",
                        "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Logging
            metrics = {}
            for l in loss:
                metrics["loss_" + l] = loss[l].item()
            for param_group in gaussians.optimizer.param_groups:
                metrics[f"lr_{param_group['name']}"] = param_group["lr"]
            training_report(
                tb_writer,
                iteration,
                metrics,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                lambda x, y: render(x, y, pipe),
                queryfunc,
            )

            if int(eval_interval) > 0 and iteration % int(eval_interval) == 0:
                vol_pred = queryfunc(scene.gaussians)["vol"]
                vol_gt = scene.vol_gt
                mse = torch.mean((vol_gt - vol_pred) ** 2).item()
                global_psnr = _psnr_from_mse(mse)
                voxel_mae = float(torch.mean(torch.abs(vol_gt - vol_pred)).item())
                voxel_mean = float(vol_pred.mean().item())
                voxel_std = float(vol_pred.std().item())
                voxel_min = float(vol_pred.min().item())
                voxel_max = float(vol_pred.max().item())
                voxel_p1 = float(torch.quantile(vol_pred, 0.01).item())
                voxel_p50 = float(torch.quantile(vol_pred, 0.50).item())
                voxel_p99 = float(torch.quantile(vol_pred, 0.99).item())
                proj_loss_val, proj_psnr_val = _projection_eval(
                    fixed_val_cameras, gaussians, pipe
                )
                density = gaussians.get_density
                scales = gaussians.get_scaling
                density_mean = float(density.mean().item())
                density_std = float(density.std().item())
                scale_mean = float(scales.mean().item())
                scale_std = float(scales.std().item())
                lr_xyz = float(
                    next(
                        pg["lr"]
                        for pg in gaussians.optimizer.param_groups
                        if pg.get("name") == "xyz"
                    )
                )
                print(
                    f"[iter {iteration}] proj_loss={proj_loss_now:.4f}  val_proj_loss={proj_loss_val:.4f}  val_proj_PSNR={proj_psnr_val:.2f} dB  global_PSNR={global_psnr:.2f} dB",
                    flush=True,
                )
                if tb_writer:
                    tb_writer.add_scalar("monitor/global_psnr", global_psnr, iteration)
                    tb_writer.add_scalar("monitor/val_proj_psnr", proj_psnr_val, iteration)
                    tb_writer.add_scalar("monitor/voxel_mae", voxel_mae, iteration)

                with open(debug_csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            iteration,
                            proj_loss_now,
                            proj_loss_val,
                            proj_psnr_val,
                            global_psnr,
                            voxel_mae,
                            voxel_mean,
                            voxel_std,
                            voxel_min,
                            voxel_max,
                            voxel_p1,
                            voxel_p50,
                            voxel_p99,
                            int(gaussians.get_density.shape[0]),
                            density_mean,
                            density_std,
                            scale_mean,
                            scale_std,
                            int(densify_added),
                            int(prune_removed),
                            lr_xyz,
                        ]
                    )

                if early_stop:
                    past_iter = iteration - 300
                    if past_iter in loss_history:
                        loss_300_ago = float(loss_history[past_iter])
                        if loss_300_ago > 0.0:
                            rel_improve = (loss_300_ago - proj_loss_now) / loss_300_ago
                        else:
                            rel_improve = float("inf")
                        if rel_improve < 0.001:
                            plateau_count += 1
                        else:
                            plateau_count = 0

                        if plateau_count >= 3:
                            print(
                                f"[EARLY STOP] at iter {iteration}, loss plateau detected",
                                flush=True,
                            )
                            scene.save(iteration, queryfunc)
                            progress_bar.close()
                            break


def training_report(
    tb_writer,
    iteration,
    metrics_train,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    queryFunc,
):
    # Add training statistics
    if tb_writer:
        for key in list(metrics_train.keys()):
            tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "train/total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    if iteration in testing_iterations:
        # Evaluate 2D rendering performance
        eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()

        validation_configs = [
            {"name": "render_train", "cameras": scene.getTrainCameras()},
            {"name": "render_test", "cameras": scene.getTestCameras()},
        ]
        psnr_2d, ssim_2d = None, None
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                images = []
                gt_images = []
                image_show_2d = []
                # Render projections
                show_idx = np.linspace(0, len(config["cameras"]), 7).astype(int)[1:-1]
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = renderFunc(
                        viewpoint,
                        scene.gaussians,
                    )["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    images.append(image)
                    gt_images.append(gt_image)
                    if tb_writer and idx in show_idx:
                        image_show_2d.append(
                            torch.from_numpy(
                                show_two_slice(
                                    gt_image[0],
                                    image[0],
                                    f"{viewpoint.image_name} gt",
                                    f"{viewpoint.image_name} render",
                                    vmin=gt_image[0].min() if iteration != 1 else None,
                                    vmax=gt_image[0].max() if iteration != 1 else None,
                                    save=True,
                                )
                            )
                        )
                images = torch.concat(images, 0).permute(1, 2, 0)
                gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
                psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
                ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
                eval_dict_2d = {
                    "psnr_2d": psnr_2d,
                    "ssim_2d": ssim_2d,
                    "psnr_2d_projs": psnr_2d_projs,
                    "ssim_2d_projs": ssim_2d_projs,
                }
                with open(
                    osp.join(eval_save_path, f"eval2d_{config['name']}.yml"),
                    "w",
                ) as f:
                    yaml.dump(
                        eval_dict_2d, f, default_flow_style=False, sort_keys=False
                    )

                if tb_writer:
                    image_show_2d = torch.from_numpy(
                        np.concatenate(image_show_2d, axis=0)
                    )[None].permute([0, 3, 1, 2])
                    tb_writer.add_images(
                        config["name"] + f"/{viewpoint.image_name}",
                        image_show_2d,
                        global_step=iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/psnr_2d", psnr_2d, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/ssim_2d", ssim_2d, iteration
                    )

        # Evaluate 3D reconstruction performance
        vol_pred = queryFunc(scene.gaussians)["vol"]
        vol_gt = scene.vol_gt
        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
        ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
        eval_dict = {
            "psnr_3d": psnr_3d,
            "ssim_3d": ssim_3d,
            "ssim_3d_x": ssim_3d_axis[0],
            "ssim_3d_y": ssim_3d_axis[1],
            "ssim_3d_z": ssim_3d_axis[2],
        }
        with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
        if tb_writer:
            image_show_3d = np.concatenate(
                [
                    show_two_slice(
                        vol_gt[..., i],
                        vol_pred[..., i],
                        f"slice {i} gt",
                        f"slice {i} pred",
                        vmin=vol_gt[..., i].min(),
                        vmax=vol_gt[..., i].max(),
                        save=True,
                    )
                    for i in np.linspace(0, vol_gt.shape[2], 7).astype(int)[1:-1]
                ],
                axis=0,
            )
            image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
            tb_writer.add_images(
                "reconstruction/slice-gt_pred_diff",
                image_show_3d,
                global_step=iteration,
            )
            tb_writer.add_scalar("reconstruction/psnr_3d", psnr_3d, iteration)
            tb_writer.add_scalar("reconstruction/ssim_3d", ssim_3d, iteration)
        tqdm.write(
            f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}"
        )

        # Record other metrics
        if tb_writer:
            tb_writer.add_histogram(
                "scene/density_histogram", scene.gaussians.get_density, iteration
            )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # fmt: off
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--early_stop", action="store_true", default=False)
    parser.add_argument("--debug_metrics_csv", type=str, default="")
    parser.add_argument("--disable_densify", action="store_true", default=False)
    parser.add_argument("--disable_prune", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    if args.max_iterations is not None:
        args.iterations = int(args.max_iterations)
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(1)
    # fmt: on

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load configuration files
    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]

    # Set up logging writer
    tb_writer = prepare_output_and_logger(args)

    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        tb_writer,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.eval_interval,
        args.early_stop,
        args.debug_metrics_csv,
        args.disable_densify,
        args.disable_prune,
    )

    # All done
    print("Training complete.")
