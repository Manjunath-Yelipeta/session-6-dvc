import os
import rootutils

# Setup the root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from src.utils.log_utils import setup_logging
from src.utils.task_wrapper import task_wrapper
from pytorch_lightning.loggers import CSVLogger

logger = setup_logging()

@task_wrapper
@hydra.main(version_base="1.3", config_path="../config", config_name="eval")
def eval(cfg: DictConfig) -> None:
    # Setup datamodule
    print(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)

    # Load model from checkpoint
    ckpt_path = cfg.ckpt_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    
    print(f"Loading model from checkpoint: {ckpt_path}")
    model_class = hydra.utils.get_class(cfg.model._target_)
    model = model_class.load_from_checkpoint(ckpt_path)

    # Setup CSV logger to log to the same directory
    csv_logger = CSVLogger(
        save_dir=cfg.paths.output_dir,
        name="eval",
        version="0",
        flush_logs_every_n_steps=1  # Add this to ensure immediate writing
    )

    # Add this after creating the logger
    print(f"Logging metrics to: {csv_logger.log_dir}/metrics.csv")

    # Setup trainer with the logger
    print(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=csv_logger,
        _convert_="partial"
    )

    # Evaluate the model
    print("Starting model evaluation")
    test_results = trainer.test(model=model, datamodule=datamodule)
    
    # Log the test results
    for metric_name, metric_value in test_results[0].items():
        print(f"{metric_name}: {metric_value}")

    # Add this line to print the accuracy in an easily parsable format
    if 'test/acc' in test_results[0]:
        print(f"Test accuracy: {test_results[0]['test/acc']:.4f}")
    else:
        print("Test accuracy not found in results")

    # If you need additional custom evaluation, you can add it here
    # For example:
    # custom_metric = custom_evaluation(model, datamodule)
    # print(f"Custom metric: {custom_metric}")

if __name__ == "__main__":
    eval()
