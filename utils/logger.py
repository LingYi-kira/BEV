import os
import logging
from datetime import datetime


def logger_setup(experiment, stage):
    
    formatter = logging.Formatter(
        fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    save_dir = "./outputs"
    title = '{}/{}'.format(experiment, stage)
    log_file_path = save_dir + "/"  + title + ".log"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    logger = logging.getLogger("Ely")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    
    return logger
    
