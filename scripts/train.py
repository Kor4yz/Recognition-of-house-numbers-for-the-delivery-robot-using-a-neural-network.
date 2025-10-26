import os, torch, pytorch_lightning as pl, torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from torch.optim import AdamW
from torchmetrics import Accuracy
from src.data.datamodule import SVHNDataModule
from src.models.resnet import SVHNResNet
from src.utils.seed import set_seed

class LitModel(LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = SVHNResNet()
        self.acc = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x): return self.net(x)

    def training_step(self, batch, _):
        x, y = batch[0], batch[1]
        logits = self(x); loss = F.cross_entropy(logits, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch[0], batch[1]
        logits = self(x); loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)
        self.acc.update(pred, y)
        self.log("val/loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("val/acc", self.acc.compute(), prog_bar=True)
        self.acc.reset()

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr)

if __name__ == "__main__":
    set_seed(42)
    dm = SVHNDataModule()
    model = LitModel(lr=1e-3)
    trainer = Trainer(max_epochs=20, accelerator="auto", devices="auto", log_every_n_steps=50)
    os.makedirs("checkpoints", exist_ok=True)
    trainer.fit(model, datamodule=dm)
    trainer.save_checkpoint("checkpoints/best.ckpt")
