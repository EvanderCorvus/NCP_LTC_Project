from utils import *
from network_utils import *
import torch as tr
import matplotlib.pyplot as plt
from networks import DeepQ_LTC_NCP
import seaborn as sns
import pytorch_lightning as pl

device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
hyperparams = Hyperparameters()
print('device: ', device)

LTC = DeepQ_LTC_NCP(hyperparams.in_features, hyperparams.hidden_dim, hyperparams.out_features)
learner = SequenceLearner(LTC, hyperparams)

train_dataloader = create_dataloader(hyperparams.n_train_samples)
csv_logger = pl.loggers.CSVLogger('logs', name='LTC')
callback = mod_Early_Stopping(hyperparams.target_mape)

trainer = pl.Trainer(devices=1,
                    accelerator= pl.accelerators.CUDAAccelerator(),
                    max_epochs=hyperparams.max_epochs,
                    gradient_clip_val=hyperparams.gradient_clip_val,                 
                    logger = csv_logger,
                    enable_progress_bar=False,
                    callbacks = [callback]
                )

trainer.fit(learner, train_dataloader)


test_dataloader = create_dataloader(hyperparams.n_test_samples)
trainer.test(learner, test_dataloader)


# Plotting
sns.set()
plt.plot(learner.losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Train Loss Over Time')
plt.savefig('losses.png')