# Dog Breed Classifier

This project implements a dog breed classifier using PyTorch Lightning. It includes scripts for training, evaluation, and inference, along with utility functions for logging and error handling.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Setup](#setup)
3. [Usage](#usage)
   - [Training](#training)
   - [Evaluation](#evaluation)
   - [Inference](#inference)
4. [Key Components](#key-components)
   - [Data Module](#data-module)
   - [Dog Classifier Model](#dog-classifier-model)
   - [Logging](#logging)
   - [Task Wrapper](#task-wrapper)
5. [Running Tests](#running-tests)
6. [Docker](#docker)
7. [DevContainer](#devcontainer)

## Project Structure

```
dog-classifier-pl/
├── src/
│   ├── datamodules/
│   │   └── dog_datamodule.py
│   ├── model/
│   │   └── dog_classifier.py
│   ├── utils/
│   │   ├── logging_utils.py
│   │   └── task_wrapper.py
│   ├── train.py
│   ├── eval.py
│   └── infer.py
├── tests/
│   ├── test_dog_datamodule.py
│   ├── test_dog_classifier.py
│   ├── test_eval.py
│   ├── test_train.py
│   ├── test_infer.py
│   ├── test_logging_utils.py
│   └── test_task_wrapper.py
├── Dockerfile
├── .devcontainer/
│   └── devcontainer.json
├── pyproject.toml
└── README.md
```

## Setup

This project uses `uv` for dependency management. To set up the project:

1. Install `uv`:
   ```
   pip install uv
   ```

2. Clone the repository:
   ```
   git clone https://github.com/yourusername/dog-breed-classifier.git
   cd dog-breed-classifier
   ```

3. Set up the Python environment and install dependencies:
   ```
   uv python install 3.12
   uv sync
   ```

## Usage

### Training

To train the model:

```
python src/train.py
```

Customize training parameters using Hydra configuration overrides:

```
python src/train.py data=dog model=timm_classify model.model_name=vit_base_patch16_224 model.num_classes=10 trainer.max_epochs=5 trainer.accelerator=cpu trainer.devices=1 data.batch_size=8
```

This script will:
- Load and preprocess the dataset
- Initialize the model
- Train the model using PyTorch Lightning
- Save the best model checkpoint based on validation loss
- Log training progress to TensorBoard

Training logs and model checkpoints will be saved in the `logs/` directory.

### Evaluation

To evaluate a trained model:

```
python src/eval.py ckpt_path=/path/to/checkpoint.ckpt
```

This script will:
- Load the specified model checkpoint
- Evaluate the model on the test dataset
- Generate and save a classification report

The classification report will be saved as `logs/classification_report.txt`.

### Inference

To run inference on new images:

```
python src/infer.py input_folder=/path/to/input/images output_folder=/path/to/output ckpt_path=/path/to/checkpoint.ckpt
```

This script will:
- Load the specified model checkpoint
- Process all images in the input folder
- Generate predictions for each image
- Save visualizations of the predictions in the output folder

## Key Components

### Data Module

The `DogDataModule` in `src/datamodules/dog_datamodule.py` handles data loading and preprocessing. It:
- Downloads and extracts the dataset if not present
- Applies data augmentation and normalization
- Creates train, validation, and test data loaders

### Dog Classifier Model

The `DogClassifier` in `src/model/dog_classifier.py` defines the model architecture and training process. It:
- Uses a pre-trained ResNet18 model from the `timm` library
- Implements the training, validation, and test steps
- Configures the optimizer and learning rate scheduler

### Logging

The `setup_logger` function in `src/utils/logging_utils.py` configures logging using Loguru. Logs are saved in the `logs/` directory.

### Task Wrapper

The `task_wrapper` decorator in `src/utils/task_wrapper.py` provides error handling and logging for main functions. It:
- Logs the start and end of each wrapped function
- Catches and logs any exceptions that occur during execution

## Running Tests

To run the test suite:

```
poetry run pytest
```

This will execute all tests in the `tests/` directory. You can run specific test files or functions using pytest's filtering options.

## Docker

To build and run the project using Docker:

1. Build the Docker image:
   ```
   docker build -t dog-classifier .
   ```

2. Run the container:
   - For training:
     ```
     docker run -v /path/to/data:/app/data -v /path/to/logs:/app/logs dog-classifier python src/train.py
     ```
   - For evaluation:
     ```
     docker run -v /path/to/logs:/app/logs dog-classifier python src/eval.py --ckpt_path /app/logs/checkpoints/best_model.ckpt
     ```
   - For inference:
     ```
     docker run -v /path/to/input:/app/input -v /path/to/output:/app/output -v /path/to/logs:/app/logs dog-classifier python src/infer.py --input_folder /app/input --output_folder /app/output --ckpt_path /app/logs/checkpoints/best_model.ckpt
     ```

## DevContainer

This project includes a DevContainer configuration for use with Visual Studio Code:

1. Install the "Remote - Containers" extension in VS Code.
2. Open the project folder in VS Code.
3. Click the green button in the lower-left corner and select "Reopen in Container".
4. VS Code will build the DevContainer and provide you with a fully configured development environment.

Within the DevContainer, you can run the training, evaluation, and inference scripts as described in the [Usage](#usage) section.

## Code Coverage

This project uses Codecov for code coverage reporting. The coverage should be at least 70%.

## Docker

The project includes a Dockerfile for containerization. The Docker image is automatically built and pushed to GitHub Container Registry (GHCR) when changes are pushed to the main branch.

To use the Docker image:

```
docker pull ghcr.io/yourusername/dog-breed-classifier:latest
docker run ghcr.io/yourusername/dog-breed-classifier:latest python src/train.py
```

## GitHub Actions

The project includes two GitHub Actions workflows:

1. CI Workflow: Runs tests, builds the Docker image, and pushes it to GHCR.
2. Train and Test Workflow: Uses the Docker image to train the model for 5 epochs and fails if the accuracy is below 95%.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
