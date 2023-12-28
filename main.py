from utils import *
from network_utils import *
import torch as tr
import matplotlib.pyplot as plt
from networks import DeepQ_LTC_NCP
import seaborn as sns
import pytorch_lightning as pl

device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
print('device: ', device)

in_features = 3
out_features = 1
hidden_dim = 256

LTC = DeepQ_LTC_NCP(in_features, hidden_dim, out_features)
learner = SequenceLearner(LTC, lr=0.005)
dataloader = create_dataloader(500)
trainer = pl.Trainer(devices=1,
                     accelerator= pl.accelerators.CUDAAccelerator(),
                    max_epochs=100,
                    gradient_clip_val=1                     
                )

trainer.fit(learner, dataloader)

trainer.test(learner, dataloader)



sns.set()
plt.plot(learner.losses)
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.savefig('losses.png')