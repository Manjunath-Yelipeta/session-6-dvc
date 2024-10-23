import pytest
import torch
from hydra.utils import instantiate
from src.eval import eval


def test_eval(eval_config):
    # Run the evaluation
    eval(eval_config)

    # Instantiate the model and datamodule
    model = instantiate(eval_config.model)
    datamodule = instantiate(eval_config.data)

    # Check if the model is properly loaded
    assert isinstance(model, torch.nn.Module)

    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    output = model(input_tensor)

    assert output.shape == (batch_size, eval_config.model.num_classes)

    # Test test step
    batch = next(iter(datamodule.test_dataloader()))
    test_step_output = model.test_step(batch, 0)

    assert isinstance(test_step_output, dict)
    assert "test_loss" in test_step_output
    assert isinstance(test_step_output["test_loss"], torch.Tensor)

    # Test the datamodule
    assert hasattr(datamodule, 'test_dataloader')
    test_loader = datamodule.test_dataloader()
    assert isinstance(next(iter(test_loader)), tuple)

    # Check if the checkpoint path exists
    import os
    assert os.path.exists(eval_config.ckpt_path), f"Checkpoint not found at {eval_config.ckpt_path}"

    # You might want to add more specific tests based on your eval function's behavior
    # For example, checking if certain metrics are computed, logged, etc.

# Add more tests as needed