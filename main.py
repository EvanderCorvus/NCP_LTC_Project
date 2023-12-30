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
hidden_dim = 128
LTC = DeepQ_LTC_NCP(in_features, hidden_dim, out_features)

learner = SequenceLearner(LTC, lr=0.005)
# DataLoaders = [create_dataloader(100) for _ in range(10)]

train_dataloader = create_dataloader(1000)
csv_logger = pl.loggers.CSVLogger('logs', name='LTC')

trainer = pl.Trainer(devices=1,
                    accelerator= pl.accelerators.CUDAAccelerator(),
                    max_epochs=100,
                    gradient_clip_val=1,                     
                    logger = csv_logger,
                    enable_progress_bar=False
                )

print('training')
# for train_dataloader in DataLoaders:    
trainer.fit(learner, train_dataloader)


print('testing')
test_dataloader = create_dataloader(250)
trainer.test(learner, test_dataloader)
print('finished')


# Plotting
sns.set()
plt.plot(learner.losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Train Loss Over Time')
plt.savefig('losses.png')