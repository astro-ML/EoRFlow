import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots
plt.style.use('science')

# Define the directory where the model files are stored
noise_model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/paper_models/noise_10_512'  
pure_model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/ps2d/oldDL/pure_ps2d_10_512_1.0'  
aaStar_model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/ps2d/oldDL/aaStar_mod_ps2d_10_512_oldDL'

# Load the training and validation losses
train_losses_noise = np.load(os.path.join(noise_model_dir, 'train_losses.npy'))
val_losses_noise = np.load(os.path.join(noise_model_dir, 'val_losses.npy'))
train_losses_pure = np.load(os.path.join(pure_model_dir, 'train_losses.npy'))
val_losses_pure = np.load(os.path.join(pure_model_dir, 'val_losses.npy'))
train_losses_aaStar = np.load(os.path.join(aaStar_model_dir, 'train_losses.npy'))
val_losses_aaStar = np.load(os.path.join(aaStar_model_dir, 'val_losses.npy'))

# Define stopping points
stop_epoch_pure = 124-1
stop_epoch_noise = 135-1
stop_epoch_aaStar = 211-1

# Plot the losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses_pure, color="#6A6B6F", linestyle='dashed', label='Noiseless Train')
plt.plot(val_losses_pure, color="#2C2E2F", label='Noiseless Val')
plt.plot(train_losses_aaStar, color='#AEC7E8', linestyle='dashed', label='AA* mod Train')
plt.plot(val_losses_aaStar, color='#1F77B4', label='AA* mod Val')
plt.plot(train_losses_noise, color='#FF9896', linestyle='dashed', label='AA4 opt Train')
plt.plot(val_losses_noise, color='#D62728', label='AA4 opt Val')

plt.scatter(stop_epoch_pure, train_losses_pure[stop_epoch_pure], color='#6A6B6F', marker='o', zorder=5)
plt.scatter(stop_epoch_pure, val_losses_pure[stop_epoch_pure], color='#2C2E2F', marker='o', zorder=5)
plt.scatter(stop_epoch_aaStar, train_losses_aaStar[stop_epoch_aaStar], color='#AEC7E8', marker='o', zorder=5)
plt.scatter(stop_epoch_aaStar, val_losses_aaStar[stop_epoch_aaStar], color='#1F77B4', marker='o', zorder=5)
plt.scatter(stop_epoch_noise, train_losses_noise[stop_epoch_noise], color='#FF9896', marker='o', zorder=5)
plt.scatter(stop_epoch_noise, val_losses_noise[stop_epoch_noise], color='#D62728', marker='o', zorder=5)

#plt.yscale('log')  
plt.xlabel('Epoch', fontsize=30)
plt.xlim(0,230)
plt.ylabel('Loss', fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=26)
plt.tick_params(axis='both', which='minor', labelsize=24)
#plt.title('Training and Validation Loss')
plt.legend(fontsize=28)
#plt.grid(True)

# Save the plot in the same model directory
plot_filename = os.path.join('/remote/gpu01a/pietschke/EoRFlow/output/paper_plots', 'loss_plot_final.pdf')
plt.savefig(plot_filename)
print(f"Loss plot saved to: {plot_filename}")


