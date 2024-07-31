import sys
sys.path.append('.')

import torch
import random
import numpy as np
import torch.utils.data as torchdata
import torchvision.transforms as tvf
import pytorch_lightning as pl
import scipy.io as sio
from typing import Any, Dict, List
from copy import deepcopy
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Optional

from utils.wrappers import Camera
from utils.torch import worker_init_fn, collate
from utils.data import read_pose_from_text, rotationError, read_image, read_calib, ToTensor, Resize


IMU_FREQ = 10

class CombinedTransform(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, imus, gts, camera, seed):
        for transform in self.transforms:
            if callable(transform):
                images, imus, gts, camera = transform(images, imus, gts, camera, seed)
            else:
                raise TypeError(f"Transform {transform} is not callable")
        return images, imus, gts, camera

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for transform in self.transforms:
            format_string += '\n    {0}'.format(transform)
        format_string += '\n)'
        return format_string


class KittiDataset(torchdata.Dataset):
    def __init__(
        self,
        stage: str,
        cfg: DictConfig,
        samples: List,
    ):
        self.stage = stage
        self.cfg = deepcopy(cfg)
        self.samples = samples
        
        tfs = []
        if stage == "train" and cfg.augmentation.image.apply:
            args = OmegaConf.masked_copy(
                cfg.augmentation.image, ["brightness", "contrast", "saturation", "hue"]
            )
            tfs.append(tvf.ColorJitter(**args))
        self.tfs = tvf.Compose(tfs)

        self.transform = CombinedTransform([ToTensor(self.tfs), Resize((cfg.img_h, cfg.img_w))])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Non fixed seed
        if self.stage == "train" and self.cfg.random:
            seed = None
        else:
            seed = [self.cfg.seed, idx]
        (seed,) = np.random.SeedSequence(seed).generate_state(1)
        
        sample = self.samples[idx]
        # (S, C, H, W)
        imgs = [read_image(img) for img in sample['imgs']]

        cam_dict = sample["cam"]
        cam = Camera.from_dict(cam_dict).float()

        if self.transform is not None:
            imgs, imus, gts, cam = self.transform(imgs, np.copy(sample['imus']), np.copy(sample['gts']), cam, seed)
        else:
            imus = np.copy(sample['imus'])
            gts = np.copy(sample['gts']).astype(np.float32)

        return {"image": imgs, 
                "imu": imus,
                "pose": gts,
                "camera": cam,
                }


class KittiDataModule(pl.LightningDataModule):
    default_cfg = {
        "name": "kitti",
        "data_dir": "./data",
        "sequence_length": 2,
        "train_seqs": ['00', '01', '02', '04', '05', '06', '07', '08'],
        "test_seqs": "???",
        "loading": {
            "train": "???",
            "val": "???",
            "test": {"batch_size": 1, "num_workers": 0},
        },
    }
    def __init__(self, cfg):
        default_cfg = OmegaConf.create(self.default_cfg)
        self.cfg = cfg = OmegaConf.merge(default_cfg, cfg)
        OmegaConf.set_struct(cfg, True)
        self.root = Path(self.cfg.data_dir)
        self.sequence_length = self.cfg.sequence_length
        self._log_hyperparams = False
        self.allow_zero_length_dataloader_with_multiple_devices = False

    def prepare_data(self):
        # No data reading and preprocessing.
        if not (self.root.exists() and (self.root / "images").exists()
                and (self.root / "imus").exists() and (self.root / "poses").exists()
                ):
            raise FileNotFoundError(
                "Cannot find the KITTI dataset, run maploc.data.kitti.prepare"
            )
    
    def prepare_data_per_node(self):
        pass
    
    def pack_data(self):
        sequence_set = []
        for folder in self.data_path:
            poses, poses_rel = read_pose_from_text(self.root/'poses/{}.txt'.format(folder))
            imus = sio.loadmat(self.root/'imus/{}.mat'.format(folder))['imu_data_interp']
            fpaths = sorted((self.root/'images/{}/image_2'.format(folder)).glob("*.png"))  # image path
            cam = read_calib(self.root/'images/{}/calib.txt'.format(folder))
            for i in range(len(fpaths)-self.sequence_length):
                img_samples = fpaths[i:i+self.sequence_length]
                imu_samples = imus[i*IMU_FREQ:(i+self.sequence_length-1)*IMU_FREQ+1]
                pose_samples = poses[i:i+self.sequence_length]
                pose_rel_samples = poses_rel[i:i+self.sequence_length-1]
                segment_rot = rotationError(pose_samples[0], pose_samples[-1])
                sample = {'imgs':img_samples, 'imus':imu_samples, 'gts': pose_rel_samples, 'rot': segment_rot, 'cam': cam}
                sequence_set.append(sample)

        train_set = val_set = test_set = []
        
        if self.stage == 'fit' or self.stage is None:
            # 定义验证集比例
            val_split = self.cfg.loading.val.val_split
            val_seed = self.cfg.seed
            val_size = int(val_split * len(sequence_set))

            random.seed(val_seed)
            random_indices = random.sample(range(len(sequence_set)), val_size)
            train_set = [sequence_set[i] for i in range(len(sequence_set)) if i not in random_indices]
            val_set = [sequence_set[i] for i in random_indices]
        elif self.stage == 'test':
            test_set = sequence_set
            
        self.samples = {"train": train_set, 
                        "val": val_set,
                        "test": test_set,
                        }
        
        
    def setup(self, stage: Optional[str] = None):
        self.stage = stage
        # Get data path
        if stage == 'fit':
            self.data_path = self.cfg.train_seqs
        elif stage == 'test':
            self.data_path = [self.cfg.test_seqs]
        elif stage is None:
            self.data_path = self.cfg.train_seqs

        self.pack_data()
    
    def dataset(self, stage: str):
        # self.setup()   # for test
        return KittiDataset(
            stage,
            self.cfg,
            self.samples[stage],
        )

    def dataloader(
            self, 
            stage: str,
            shuffle: bool = False,
            num_workers: int = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers

        loader = torchdata.DataLoader(
            dataset,
            batch_size = cfg["batch_size"],
            shuffle = shuffle or (stage == "train"),
            num_workers = num_workers,
            pin_memory = True,
            persistent_workers = num_workers > 0,
            worker_init_fn = worker_init_fn,
            collate_fn = collate,  # 处理复杂的批次组合逻辑
            )
        
        return loader
    
    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)


        
