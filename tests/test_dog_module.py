import pytest
from hydra.utils import instantiate

def test_datamodule(train_config):
    try:
        datamodule = instantiate(train_config.data)
    except Exception as e:
        print(f"Full error: {e}")
        raise

    datamodule.setup()

    assert len(datamodule.train_dataset) > 0
    assert len(datamodule.val_dataset) > 0
    assert len(datamodule.test_dataset) > 0

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0

    batch = next(iter(train_loader))
    assert len(batch) == 2  # (images, labels)
    assert batch[0].shape[1:] == (3, 224, 224)  # (batch_size, channels, height, width)