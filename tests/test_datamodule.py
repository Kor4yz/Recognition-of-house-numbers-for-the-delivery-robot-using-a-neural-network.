from src.data.datamodule import SVHNDataModule

def test_dm_loads():
    dm = SVHNDataModule(batch_size=8, num_workers=0)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert x.shape[0] == 8
    assert x.shape[-1] == 32 and x.shape[-2] == 32
    assert y.ndim == 1
