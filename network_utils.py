import numpy as np
import torch as tr
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError
from pytorch_lightning import Callback


class SequenceLearner(pl.LightningModule):
    def __init__(self, model, hyperparams):
        super().__init__()
        self.model = model
        self.lr = hyperparams.learning_rate
        self.step_size = hyperparams.step_size
        self.gamma = hyperparams.gamma

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
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        #rmse = self.rmse(y_hat, y)
        mape = self.mape(y_hat, y)
        self.log('val_mape', mape, on_step=False, on_epoch=True, logger=True)
        return mape
    

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
        scheduler = tr.optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=self.step_size,
                                                  gamma=self.gamma)
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',    # will update after every epoch...
            'frequency': 1          # ...with this frequency
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}
 

class EarlyStopping(Callback):
    def __init__(self, target_mape):
        super().__init__()
        self.target_mape = target_mape

    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Called at the end of the validation epoch.
        """
        val_mape = trainer.callback_metrics.get("val_mape")
        if val_mape < self.target_mape:
            print(f"Stopping training as validation MAPE has reached {val_mape}")
            trainer.should_stop = True
