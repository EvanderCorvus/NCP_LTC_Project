import numpy as np
import torch as tr
import torch.nn as nn
import pytorch_lightning as pl
from ncps.torch import LTC
from ncps.wirings import AutoNCP, NCP

class DeepQ_LTC_NCP(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, out_dim = 1, num_layers = 2):
        super(DeepQ_LTC_NCP, self).__init__()

        wiring = AutoNCP(hidden_dim,
                        out_features
                        )
        self.rnn = LTC(in_features, wiring,
                       batch_first=True
                       )
        
        #hidden represents both ltc and fc hidden dim!
        #self.fc = nn.Linear(out_features, out_dim)
                            
        
    def forward(self, state):
        x = self.rnn(state)
       # x = self.fc(state)
        return x
