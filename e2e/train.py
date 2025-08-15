import pytorch_lightning as pl
import torch

from datamodule import VideoDataModule
from model_wrapper import CustomModel

torch.set_float32_matmul_precision("medium")

if __name__ == '__main__':
    data_dir = '/work/m24oc/m24oc/s1852485/Diss/train/e2e/dataset'

    batch_size = 16
    num_workers = 10
    channels = 64
    model_depth = 6
    B = 20

    vdm = VideoDataModule(data_dir, batch_size=batch_size,
                          num_workers=num_workers, B=B)

    model = CustomModel(B=B, channels=channels, depth=model_depth)

    logger = pl.loggers.TensorBoardLogger(
        save_dir="lightning_logs", log_graph=False,
        name="E2E-CNN", version="v0"
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=5)

    trainer = pl.Trainer(logger=logger, callbacks=[early_stop_callback],
                         min_epochs=10, max_epochs=100,
                         accelerator="gpu", devices=2, num_nodes=1,strategy="ddp"
                         )

    trainer.fit(model, vdm)

    trainer.test(model, vdm)
