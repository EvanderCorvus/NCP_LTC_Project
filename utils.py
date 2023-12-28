import configparser
import numpy as np
from deap import base, creator, tools
import torch as tr
import torch.utils.data as data

# Currently only works for floats, ints and lists
# Beware PATH!
class Hyperparameters:
    def __init__(self, config_file = 'hyperparameters.ini'):
        config = configparser.ConfigParser()
        config.read(config_file)
        if not config.read(config_file):
            raise Exception("Could not read config file")

        for key in config['Hyperparameters']:
            try:
                value = int(config['Hyperparameters'][key])
            except ValueError:
                try:
                    value = float(config['Hyperparameters'][key])
                except ValueError:
                    try:
                        value = np.array(list(config['Hyperparameters'][key]))
                    except ValueError:
                        value = config['Hyperparameters'][key]
            setattr(self, key, value)

class time:
    def __init__(self, dt, episode_length, device):
        self.t = tr.tensor(0).to(device)
        self.dt = tr.tensor(dt).to(device)
        self.episode_length = episode_length

# Create Dataloader for $y: [-1,1]^{2}\rightarrow \mathbb{R}\quad y(x,t) = [\sin(\omega t + k\sqrt{x[1]^2}, 0]$
        
def create_dataloader(n_samples, omega = 1, k=1, T = 10):
    t = np.array([np.linspace(0,T,n_samples)]).T
    x = np.random.rand(n_samples, 2)
    
    x = np.append(x, t, axis = 1)
    y = np.sin(omega*x[:,2] + k*np.sqrt(x[:,1]**2))

    data_x = tr.tensor(x).float()
    data_y = tr.tensor(y).float()
    
    dataset = data.TensorDataset(data_x, data_y)
    dataloader = data.DataLoader(dataset, batch_size = 1, 
                                shuffle = True
                                )
    return dataloader
