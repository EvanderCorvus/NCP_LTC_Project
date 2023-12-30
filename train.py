from utils import *
from network_utils import *
import torch as tr
import matplotlib.pyplot as plt
from networks import DeepQ_LTC_NCP
import seaborn as sns
import pytorch_lightning as pl


device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
checkpoint_path = '/home/weinmann/NCP_LTC_NNs/logs/LTC/version_5/checkpoints/epoch=99-step=50000.ckpt'

in_features = 3
out_features = 1
hidden_dim = 256
LTC = DeepQ_LTC_NCP(in_features, hidden_dim, out_features)
LTC.load_state_dict(tr.load(checkpoint_path)['state_dict'])

learner = SequenceLearner(LTC)
test_dataloader = create_dataloader(10)
csv_logger = pl.loggers.CSVLogger('logs', name='LTC')

trainer = pl.Trainer(devices=1,
                    accelerator= pl.accelerators.CUDAAccelerator(),
                    max_epochs=100,
                    gradient_clip_val=1,                     
                    logger = csv_logger,
                    enable_progress_bar=False
                )


trainer.test(learner, test_dataloader)
print('finished')