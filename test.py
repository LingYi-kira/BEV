import hydra
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import pytorch_lightning as pl
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from data.dataset import KittiDataset, KittiDataModule
from utils.data import read_pose_from_text, rotationError, read_image, read_calib, ToTensor, Resize
from scipy.spatial.transform import Rotation as R
# from module import GenericModule

import random

def test_data(dataset):
    # 遍历整个数据集以检查所有样本
    random_indices = random.sample(range(len(dataset)), 10)
    print(len(dataset))

    for i in random_indices:
        # data = dataset[i]
        try:
            data = dataset[i]
            # print(data["pose"].size())
        except Exception as e:
            print(f"Error in sample {i}: {e}")

def read_calib(path):
    with open(path) as f:
        lines = [line.split('\n')[0] for line in f.readlines()]
        p2_values = lines[1].split()[1:]
        p2_values = list(map(float, p2_values))
        params_K = np.array([p2_values[0], p2_values[5], p2_values[2], p2_values[6]])

    return {'model': 'PINHOLE', 
             'width': 1242, 
             'height': 375, 
             'params': params_K
    }


def make_grid(
    w: float,
    h: float,
    step_x: float = 1.0,
    step_y: float = 1.0,
    orig_x: float = 0,
    orig_y: float = 0,
    y_up: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    x, y = torch.meshgrid(
        [
            torch.arange(orig_x, w + orig_x, step_x, device=device),
            torch.arange(orig_y, h + orig_y, step_y, device=device),
        ],
        indexing="xy",
    )
    if y_up:
        y = y.flip(-2)
    grid = torch.stack((x, y), -1)
    return grid

def test_grid() -> None:
    x_max = z_max = 32
    Δ = 0.5

    # 生成 XZ 网格
    grid_xz = make_grid(
        x_max * 2 + Δ, z_max, step_y=Δ, step_x=Δ, orig_y=Δ, orig_x=-x_max, y_up=True
    )
    print(grid_xz.shape)

def euler2rot(euler):
    r = R.from_euler('xyz', euler, degrees=False)
    rotation_matrix = r.as_matrix()
    return rotation_matrix
def euler2quaternion(euler):
    r = R.from_euler('xyz', euler, degrees=False)
    quaternion = r.as_quat()
    return quaternion

import quaternion
# def euler_to_quaternion(euler):
#     pitch, yaw, roll = euler
#     q = quaternion.from_euler_angles(pitch, yaw, roll)
#     return q

def euler_to_quaternion2(euler):
    roll, pitch, yaw = euler
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q = np.quaternion(
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        sy * cp * sr + cy * sp * cr,
        sy * cp * cr - cy * sp * sr
    )
    return q

import torch

def euler_to_quaternion1(tensor):
    """
    Convert Euler Angles to Quaternion.
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
    :return: A tensor of shape (4,) representing the quaternion [w, x, y, z].
    """
    roll, pitch, yaw = tensor[0], tensor[1], tensor[2]
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return torch.stack([qw, qx, qy, qz])

# Example usage:

def test_kitti() -> None:
    poses, poses_rel = read_pose_from_text('./data/poses/{}.txt'.format("00"))
    # print(poses_rel[0: 2])

    print(euler2quaternion(poses_rel[0, 0:3]))
    print(euler_to_quaternion2(poses_rel[0, 0:3]))
    print(euler_to_quaternion1(torch.tensor(poses_rel[0, 0:3])))

    # q_mat1 = euler2quaternion(poses_rel[0, 0:3])
    # q_mat2 = euler2quaternion(poses_rel[10, 0:3])
    q_mat1 = euler2quaternion(torch.tensor(poses_rel[0, 0:3]))
    q_mat2 = euler2quaternion(torch.tensor(poses_rel[10, 0:3]))

	#归一化
    q1 = q_mat1 / np.linalg.norm(q_mat1)  
    q2 = q_mat2 / np.linalg.norm(q_mat2)
    #计算角度
    d = abs(np.sum(np.multiply(q1, q2)))
    d = 1.0 if d > 1.0 else -1.0 if d < -1.0 else d
    theta = 2 * np.arccos(d) * 180 / np.pi   #两个单位四元数的角度差
    print(theta)
    return theta

class ECAttention(nn.Module):
    def __init__(self, k_size=3):
        super(ECAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)        
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)

def test_eca() -> None:
    test_data = torch.rand((4, 3, 8, 8))
    atten = ECAttention()
    out = atten(test_data)


def resolve_checkpoint_path(experiment_or_path: str) -> Path:
    root_path = Path(experiment_or_path)
    path = "outputs/model" / root_path

    if path.is_file():
        return path

    maybe_path = "outputs/model" / root_path / "last.ckpt"
    if not maybe_path.exists():
        raise FileNotFoundError(f"Could not find any checkpoint in {path}.")
    
    return maybe_path


import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

# 示例用法
if __name__ == "__main__":
    input_tensor = torch.randn(1, 64, 32, 32)  # 假设输入是一个批次，64个通道，32x32的图像
    cbam = CBAM(in_planes=64)
    output_tensor = cbam(input_tensor)
    print(output_tensor.shape)  # 应输出 torch.Size([1, 64, 32, 32])




# default_cfg = OmegaConf.create(
#     {
#         "img_h": 256,
#         "img_w": 512,
#         "seed": 42
#     }
# )


# read config file
# @hydra.main(version_base=None, 
#             config_path=osp.join(osp.dirname(__file__), "conf"), config_name="default"
# )
# def main(cfg: DictConfig) -> None:
#     torch.set_float32_matmul_precision("medium")
#     OmegaConf.resolve(cfg)

    # model = GenericModule(cfg)
    
    # stage = "test"
    # stage = "train"
    # conf = OmegaConf.merge(default_cfg, OmegaConf.create({}))
    # dataset = KittiDataModule(conf)
    
    # dataset.setup("test" if stage == "test" else None)
    # train_dataset = dataset.dataset(stage)
    # test_data(train_dataset)
    
    # trainer = pl.Trainer()
    # trainer.fit(model, dataset)

    

