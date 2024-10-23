import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from rich.progress import track
from hydra.utils import instantiate
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import Callback

from src.utils.task_wrapper import task_wrapper
from src.model.timm_classifier import TimmClassifier
from src.datamodules.dog_datamodule import DataModule  # Added import for DataModule

@task_wrapper
def infer(cfg: DictConfig) -> None:
    # Setup logger
    logger: Logger = instantiate(cfg.logger)

    # Setup callbacks
    callbacks: list[Callback] = instantiate(cfg.callbacks)

    # Load the model
    model = TimmClassifier.load_from_checkpoint(cfg.ckpt_path)
    model.eval()

    # Initialize DataModule to get class labels
    datamodule: DataModule = instantiate(cfg.data)
    datamodule.setup()
    class_labels = datamodule.train_dataset.dataset.classes

    # Prepare transforms
    transform = transforms.Compose([
        transforms.Resize((cfg.inference.image_size, cfg.inference.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.inference.normalization_mean, cfg.inference.normalization_std)
    ])

    # Create output folder if it doesn't exist
    output_path = Path(cfg.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all image files
    input_path = Path(cfg.input_folder)
    image_files = list(input_path.glob('*.[pj][np][g]'))

    # Process each image
    for image_file in track(image_files, description="Processing images"):
        # Load and preprocess the image
        img = Image.open(image_file).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        # Get predicted label
        predicted_label = class_labels[predicted_class]

        # Create and save the plot
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Predicted: {predicted_label} (Confidence: {confidence:.2f})")
        
        # Create output filename with predicted class
        output_file = output_path / f"{image_file.stem}_{predicted_label}_prediction.png"
        plt.savefig(output_file)
        plt.close()

        # Log the prediction
        if hasattr(logger, 'log_metrics'):
            logger.log_metrics({
                "predicted_class": predicted_class,
                "confidence": confidence
            })
        else:
            print("Warning: logger.log_metrics not available. Skipping metric logging.")

    # Finalize the logger
    if hasattr(logger, 'finalize'):
        logger.finalize("success")
    else:
        import logging
        logging.info("Inference completed successfully")

@hydra.main(version_base=None, config_path="../config", config_name="infer")
def main(cfg: DictConfig):
    print(f"Checkpoint path: {cfg.ckpt_path}")
    infer(cfg)

if __name__ == "__main__":
    main()
