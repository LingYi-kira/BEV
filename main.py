import os
import os.path as osp
import hydra
import logging
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from module import GenericModule
from data.dataset import KittiDataModule
from utils.logger import logger_setup
from utils.callback import SeedingCallback, CleanProgressBar, ConsoleLogger


# read config file
@hydra.main(version_base=None, 
            config_path=osp.join(osp.dirname(__file__), "conf"), config_name="default"
)
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("medium")
    OmegaConf.resolve(cfg)
    rank = rank_zero_only.rank
    
    logger = logger_setup(cfg.experiment.name, "training")
    logging.root = logger

    if rank == 0:
        logger.info("Starting training with config:\n%s", OmegaConf.to_yaml(cfg))
    if cfg.experiment.gpus in (None, 0):
        logger.warning("Will train on CPU...")
        cfg.experiment.gpus = 0
    elif not torch.cuda.is_available():
        raise ValueError("Requested GPU but no NVIDIA drivers found.")
    pl.seed_everything(cfg.experiment.seed, workers=True)

    model = GenericModule(cfg)
    if rank == 0:
        logger.info("Network:\n%s", model.model)  # 记录模型的结构信息

    if cfg.experiment.gpus > 1:
        strategy = pl.strategies.DDPStrategy(find_unused_parameters=True)  # False 7.31日修改True
        for stage in ["train", "val"]:
            cfg.data["loading"][stage].batch_size = (
                cfg.data["loading"][stage].batch_size // cfg.experiment.gpus
            )
            cfg.data["loading"][stage].num_workers = int(
                (cfg.data["loading"][stage].num_workers + cfg.experiment.gpus - 1)
                / cfg.experiment.gpus
            )
        
        devices = cfg['experiment']['gpus_device']
        if len(devices) != cfg.experiment.gpus:
            devices = cfg.experiment.gpus
    else:
        strategy = "auto"

    if cfg.data.name == "kitti":
        data = KittiDataModule(cfg.data)
    else:
        raise ValueError("Inaccurate dataset name.")
    
    save_dir = osp.join(cfg.experiment.save_dir, cfg.experiment.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tensorboard_dir = save_dir + "/tensorboard"
    experiment_dir = save_dir + "/model"

    tensorboard_args = {"version": ""}
    tensorboard = pl.loggers.TensorBoardLogger(tensorboard_dir, **tensorboard_args)

    checkpointing_epoch = pl.callbacks.ModelCheckpoint(
        dirpath=experiment_dir,             # 保存检查点文件的目录
        filename="checkpoint-{epoch:02d}",  # 指定检查点文件的命名格式
        save_last=True,                     # 保存最新的模型检查点
        every_n_epochs=1,                   # 每训练n个epoch就保存一次检查点
        save_on_train_epoch_end=True,       # 在训练 epoch 结束时保存检查点。
        verbose=True,                       # 在控制台输出保存检查点的详细信息
        **cfg.training.checkpointing,
    )
    
    # checkpointing_step = pl.callbacks.ModelCheckpoint(
    #     dirpath=experiment_dir,
    #     filename="checkpoint-{step}",
    #     save_last=True,
    #     every_n_train_steps=100,
    #     verbose=True,
    #     **cfg.training.checkpointing,
    # )
    # checkpointing_step.CHECKPOINT_NAME_LAST = "last-step"

    callbacks = [
        checkpointing_epoch,
        # checkpointing_step,
        pl.callbacks.LearningRateMonitor(),
        SeedingCallback(),
        CleanProgressBar(),
        ConsoleLogger(),
    ]
    if cfg.experiment.gpus > 0:
        callbacks.append(pl.callbacks.DeviceStatsMonitor())

    trainer = pl.Trainer(
        default_root_dir=experiment_dir,
        detect_anomaly=False,
        enable_model_summary=False,
        sync_batchnorm=True,
        enable_checkpointing=True,
        logger=tensorboard,
        callbacks=callbacks,
        strategy=strategy,
        devices=devices,
        accelerator="gpu",
        num_nodes=1,
        num_sanity_val_steps=1,
        **cfg.training.trainer,
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    main()