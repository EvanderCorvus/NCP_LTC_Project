import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()
version = 16
# Load the CSV file
log_file = '/home/weinmann/NCP_LTC_NNs/logs/LTC/version_{}/metrics.csv'.format(version)

data = pd.read_csv(log_file)
print(data.keys())
# Plotting
fig, axs = plt.subplots(1, 1, figsize=(12, 6))

axs.plot(data['step'], data['train_loss'], label='Train Loss')
axs.set_xlabel('Step')
axs.set_ylabel('Loss')
axs.set_title('Train Loss Over Time in version {}'.format(version))
axs.legend()

plt.savefig('losses.png')