import numpy as np
import torch as tr
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError

class SequenceLearner(pl.LightningModule):
    def __init__(self, model, lr=0.005):
        super().__init__()
        self.model = model
        self.lr = lr
        self.losses = []
        #RMSE as testing metric
        self.rmse = MeanSquaredError(squared=False)

    def mape(self, y_pred, y_true):
        "Calculate Mean Absolute Percentage Error"
        return tr.mean(tr.abs((y_true - y_pred) / y_true)) * 100

    def training_step(self, batch):
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = nn.MSELoss()(y_hat, y)
        self.losses.append(loss.item())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch):
        # x, y = batch
        # y_hat, _ = self.model.forward(x)
        # y_hat = y_hat.view_as(y)
        # rmse = self.rmse(y_hat, y)        
        # return rmse
        pass
    
    def test_step(self, batch):
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        #rmse = self.rmse(y_hat, y)
        mape = self.mape(y_hat, y)
        self.log('Mean Absolute Percentage Error', mape)
        return mape

    def configure_optimizers(self):
        optimizer = tr.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = tr.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',    # will update after every epoch...
            'frequency': 1          # ...with this frequency
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}
 