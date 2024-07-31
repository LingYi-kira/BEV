import logging
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

logger = logging.getLogger(__name__)
pl_logger = logging.getLogger("pytorch_lightning")

class CleanProgressBar(pl.callbacks.TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.leave = False  # 训练完成后不保留进度条
        bar.mininterval = 1.0  # 最小的刷新间隔时间
        bar.position = 0   # 进度条在同一行刷新
        bar.ncols = 100
        return bar
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        items.pop("loss", None)
        return items


class SeedingCallback(pl.callbacks.Callback):
    def on_epoch_start_(self, trainer, module):
        seed = module.cfg.experiment.seed
        is_overfit = module.cfg.training.trainer.get("overfit_batches", 0) > 0
        if trainer.training and not is_overfit:
            seed = seed + trainer.current_epoch

        # Temporarily disable the logging (does not seem to work?)
        pl_logger.disabled = True
        try:
            pl.seed_everything(seed, workers=True)
        finally:
            pl_logger.disabled = False

    def on_train_epoch_start(self, *args, **kwargs):
        self.on_epoch_start_(*args, **kwargs)

    def on_validation_epoch_start(self, *args, **kwargs):
        self.on_epoch_start_(*args, **kwargs)

    def on_test_epoch_start(self, *args, **kwargs):
        self.on_epoch_start_(*args, **kwargs)


class ConsoleLogger(pl.callbacks.Callback):
    @rank_zero_only
    def on_train_epoch_start(self, trainer, module):
        logger.info(
            "New training epoch %d for experiment '%s'.",
            module.current_epoch,
            module.cfg.experiment.name,
        )

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, module):
        results = {
            **dict(module.metrics_val.items()),
            **dict(module.losses_val.items()),
        }
        results = [f"{k} {v.compute():.3E}" for k, v in results.items()]
        logger.info(f'[Validation] {{{", ".join(results)}}}')