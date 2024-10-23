import os
import sys
from functools import wraps
import torch
from torchviz import make_dot
from PIL import Image
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import Callback
from typing import List

def visualize_model(model):
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render("model_architecture", format="png")
    Image.open("model_architecture.png").show()

def setup_logging():
    log_folder = "logs"
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, "model.log")
    
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
    logger.add(log_file, rotation="10 MB", level="DEBUG")
    
    return logger

def setup_logger(log_file):
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.add(log_file, rotation="10 MB")

def task_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = setup_logging()
        try:
            logger.info(f"Starting task: {func.__name__}")
            result = func(*args, **kwargs)
            logger.info(f"Task completed: {func.__name__}")
            return result
        except Exception as e:
            logger.exception(f"Error in task {func.__name__}: {str(e)}")
            raise
    return wrapper

def get_rich_progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    )

def log_metrics_table(metrics: dict, title: str):
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    rich_print(table)

def finish(
    config: DictConfig,
    model: torch.nn.Module,
    datamodule: object,
    trainer: Trainer,
    callbacks: List[Callback],
    logger: List[Logger],
):
    """
    Makes sure everything closed properly.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        model (torch.nn.Module): Model instance.
        datamodule (object): The data module used in the experiment.
        trainer (Trainer): Trainer instance.
        callbacks (List[Callback]): List of callbacks used in the experiment.
        logger (List[
    """


