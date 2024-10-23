import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

#@pytest.fixture(scope="session")
#def train_config():
    #with initialize(version_base=None, config_path="../config"):
      #  cfg = compose(config_name="train")
    #return cfg

#@pytest.fixture(scope="session")
#def eval_config():
 #   with initialize(version_base=None, config_path="../config"):
  #      cfg = compose(config_name="eval")
   # return cfg

#@pytest.fixture(scope="session")
#def infer_config():
 #   with initialize(version_base=None, config_path="../config"):
  #      cfg = compose(config_name="infer")
   # return cfg