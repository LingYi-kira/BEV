import os
import random
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from itertools import islice
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from evaluation.utils import *


def resolve_checkpoint_path(experiment_or_path: str) -> Path:
    root_path = Path(experiment_or_path)
    path = "outputs" / root_path / "model"

    if path.is_file():
        return path

    maybe_path = "outputs" / root_path / "model/last.ckpt"
    if not maybe_path.exists():
        raise FileNotFoundError(f"Could not find any checkpoint in {path}.")
    
    return maybe_path
def test_one_path(model, dataloader):
    pose_list = []
    gt_list = []

    num = len(dataloader)
    for i, batch_ in enumerate(
        islice(tqdm(dataloader, total=num, smoothing=0.85), num)
    ):
        batch = model.transfer_batch_to_device(batch_, model.device, i)
        with torch.no_grad():
            pred = model(batch)
        
        pose_list.append(pred.detach().cpu().numpy())
        gt_list.append(batch["pose"].squeeze(dim=1).detach().cpu().numpy())
    
    pose_est = np.vstack(pose_list)
    pose_gt = np.vstack(gt_list)
    
    return pose_est, pose_gt


def kitti_eval(pose_est, pose_gt):
    
    # Calculate the translational and rotational RMSE
    t_rmse, r_rmse = rmse_err_cal(pose_est, pose_gt)

    # Transfer to 3x4 pose matrix
    pose_est_mat = path_accu(pose_est)
    pose_gt_mat = path_accu(pose_gt)

    # Using KITTI metric
    err_list, t_rel, r_rel, speed = kitti_err_cal(pose_est_mat, pose_gt_mat)
    
    t_rel = t_rel * 100
    r_rel = r_rel / np.pi * 180 * 100
    r_rmse = r_rmse / np.pi * 180

    return pose_est_mat, pose_gt_mat, t_rel, r_rel, t_rmse, r_rmse, speed


def generate_plots(seq_name, save_dir, est):
    plotPath_2D(seq_name, 
                save_dir,
                est['pose_gt_global'], 
                est['pose_est_global'], 
                est['speed'], 
                )


def evaluate(
    experiment: str,
    cfg: DictConfig,
    model,
    dataset,
    logger,
    num_workers: int = 1,
):
    
    logger.info("testing sequence %s", cfg.test_seqs)
    dataset.setup("test")
    seed_everything(dataset.cfg.seed)
    dataloader = dataset.dataloader("test", shuffle=False, num_workers=num_workers)

    pose_est, pose_gt = test_one_path(model, dataloader)
    pose_est_global, pose_gt_global, t_rel, r_rel, t_rmse, r_rmse, speed = kitti_eval(pose_est, pose_gt)
    
    est = {'pose_est_global':pose_est_global, 'pose_gt_global':pose_gt_global, 'speed':speed}
    errors = {'t_rel':t_rel, 'r_rel':r_rel, 't_rmse':t_rmse, 'r_rmse':r_rmse}
    
    full_path = os.path.join(cfg.output_dir, experiment) + "/evaluation/path"
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        
    generate_plots(cfg.test_seqs, full_path, est)
    save_text(cfg.test_seqs, full_path, est)
    
    message = f"Seq: {cfg.test_seqs}, t_rel: {errors['t_rel']:.4f}, r_rel: {errors['r_rel']:.4f}, "
    message += f"t_rmse: {errors['t_rmse']:.4f}, r_rmse: {errors['r_rmse']:.4f}, "
    logger.info(message)

def visualize_bev(
    experiment: str,
    cfg: DictConfig,
    model,
    dataset,
    logger,
    num_workers: int = 1,
):
    dataset.setup("test")
    seed_everything(dataset.cfg.seed)
    dataloader = dataset.dataloader("test", shuffle=False, num_workers=num_workers)
    
    full_path = os.path.join(cfg.output_dir, experiment) + "/evaluation/feature_map"
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    
    num = len(dataloader)
    random_indices = random.sample(range(num), 3)
    
    for i, batch_ in enumerate(
        islice(dataloader, num)
    ):
        if i in random_indices or i == 0:
            batch = model.transfer_batch_to_device(batch_, model.device, i)
            
            logger.info("visualize the bev feature map of image %s", i)
            with torch.no_grad():
                pred_bev = model.pred_bev(batch)
            
            plotBEV_FeatureMap(full_path, i, pred_bev)
            plotKitti_image(full_path, i, batch["image"][:, 0, :, :].squeeze(dim=1))
            
    
    
