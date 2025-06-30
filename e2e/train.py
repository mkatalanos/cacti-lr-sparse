import pytorch_lightning as pl
import torch

from datamodule import VideoDataModule
from model_wrapper import Model

torch.set_float32_matmul_precision("medium")

if __name__ == '__main__':
    data_dir = '/home/marios/Documents/diss-code/repo/e2e/dataset'

    batch_size = 4
    num_workers = 31
    channels = 64
    model_depth = 6
    B = 20

    vdm = VideoDataModule(data_dir, batch_size=batch_size,
                          num_workers=num_workers, B=B)

    model = Model(B=B, channels=channels, depth=model_depth)

    logger = pl.loggers.TensorBoardLogger(
        save_dir="lightning_logs", log_graph=False,
        name="E2E-CNN", version="v0"
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=5)

    trainer = pl.Trainer(logger=logger, callbacks=[
                         early_stop_callback], min_epochs=10)

    trainer.fit(model, vdm)

    trainer.test(model, vdm)
