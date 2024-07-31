import torch
import argparse
from pathlib import Path
from typing import Optional, Tuple
from omegaconf import DictConfig, OmegaConf

from module import GenericModule
from utils.logger import logger_setup
from data.dataset import KittiDataModule
from evaluation.evaluate import evaluate, resolve_checkpoint_path, visualize_bev

default_cfg = OmegaConf.create(
    {
        "img_h": 256,
        "img_w": 512,
        "seed": 42,
        "test_seqs": "???",
        "output_dir": "./outputs",
    }
)

def main(
    args, 
    cfg: Optional[DictConfig] = None,
):
    logger = logger_setup(args.experiment, "evaluation")
    logger.info("---------start evaluation---------")
    
    experiment = args.experiment
    logger.info("Evaluating model %s with config", experiment)
    checkpoint_path = resolve_checkpoint_path(experiment)
    
    model = GenericModule.load_from_checkpoint(
        checkpoint_path, logger=logger, cfg=cfg, find_best=not experiment.endswith(".ckpt")
    )
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.eval()
    
    for i in range(len(args.test_seqs)):
        cfg = OmegaConf.create({"test_seqs": args.test_seqs[i],})
        cfg = OmegaConf.merge(default_cfg, cfg)
        
        dataset = KittiDataModule(cfg)
        
        if args.visualize_bev == "True":
            visualize_bev(experiment, cfg, model, dataset, logger)
            return
        
        evaluate(experiment, cfg, model, dataset, logger)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--test_seqs", type=str, required=True, nargs='+')
    parser.add_argument('--cuda_device', default=0, type=int)
    parser.add_argument("--visualize_bev", default=False, type=str)
    
    args = parser.parse_args()
    
    main(args)
    
    

    