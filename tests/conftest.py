import os
import sys
from pathlib import Path

import pytest
from hydra import initialize, compose

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(autouse=True)
def set_project_root():
       os.environ['PROJECT_ROOT'] = '../'

@pytest.fixture
def train_config():
    with initialize(config_path="../config"):
        cfg = compose(config_name="train")
    return cfg

@pytest.fixture
def eval_config():
    with initialize(config_path="../config"):
        cfg = compose(config_name="eval")
    return cfg
