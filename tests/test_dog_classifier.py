from hydra.utils import instantiate
import torch
import pytest

def test_classifier(train_config):
    model = instantiate(train_config.model)

    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    output = model(input_tensor)

    assert output.shape == (batch_size, train_config.model.num_classes)

    # Test training step
    batch = (input_tensor, torch.randint(0, train_config.model.num_classes, (batch_size,)))
    loss = model.training_step(batch, 0)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()

    # Test validation step
    val_loss = model.validation_step(batch, 0)

    assert isinstance(val_loss, torch.Tensor)
    assert val_loss.shape == ()

    # Test test step
    model.test_step(batch, 0)

    # Test configure_optimizers
    optimizers = model.configure_optimizers()

    assert isinstance(optimizers, dict)
    assert "optimizer" in optimizers
    assert "lr_scheduler" in optimizers