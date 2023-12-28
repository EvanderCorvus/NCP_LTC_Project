import numpy as np
import torch as tr
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError
from torchmetrics.functional import mean_squared_error

class SequenceLearner(pl.LightningModule):
    def __init__(self, model, lr=0.005):
        super().__init__()
        self.model = model
        self.lr = lr
        self.losses = []
        #RMSE as testing metric
        self.mse = MeanSquaredError(squared=False)

    def training_step(self, batch):
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = nn.MSELoss()(y_hat, y)
        self.losses.append(loss.item())
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        rse = self.mse(y_hat, y)
        self.log('test_rse', rse, onstep=True, on_epoch=False)
        

        #self.log("val_loss", loss, prog_bar=True)
        return rse

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return tr.optim.Adam(self.model.parameters(), lr=self.lr)       

        



