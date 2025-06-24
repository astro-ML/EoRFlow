import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots
plt.style.use('science')

# Define the directory where the model files are stored
noise_model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/EoR_flow_logit/10_512_noise_wd-4_logit'  # Update this to your actual directory
pure_model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/EoR_flow_logit/pure_z12_10_512_-5_extraNoise'  # Update this to your actual directory

# Load the training and validation losses
train_losses_noise = np.load(os.path.join(noise_model_dir, 'train_losses.npy'))
val_losses_noise = np.load(os.path.join(noise_model_dir, 'val_losses.npy'))
train_losses_pure = np.load(os.path.join(pure_model_dir, 'train_losses.npy'))
val_losses_pure = np.load(os.path.join(pure_model_dir, 'val_losses.npy'))

outlier_index = np.argmax(train_losses_pure)  # finds the most negative (smallest) value

# Step 2: Compute mean of neighbors
if 0 < outlier_index < len(train_losses_pure) - 1:
    neighbors_mean = (train_losses_pure[outlier_index - 1] + train_losses_pure[outlier_index + 1]) / 2
elif outlier_index == 0:
    neighbors_mean = train_losses_pure[1]  # only right neighbor
elif outlier_index == len(train_losses_pure) - 1:
    neighbors_mean = train_losses_pure[-2]  # only left neighbor

# Step 3: Replace outlier
train_losses_pure[outlier_index] = neighbors_mean

# Define stopping points
stop_epoch_pure = 127-1
stop_epoch_noise = 135-1

# Plot the losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses_noise, color='#FF9896', linestyle='dashed', label='Mock Training')
plt.plot(val_losses_noise, color='#D62728', label='Mock Validation')
plt.scatter(stop_epoch_noise, train_losses_noise[stop_epoch_noise], color='#FF9896', marker='o', zorder=5)
plt.scatter(stop_epoch_noise, val_losses_noise[stop_epoch_noise], color='#D62728', marker='o', zorder=5)
plt.plot(train_losses_pure, color='#AEC7E8', linestyle='dashed', label='Noiseless Training')
plt.plot(val_losses_pure, color='#1F77B4', label='Noiseless Validation')
plt.scatter(stop_epoch_pure, train_losses_pure[stop_epoch_pure], color='#AEC7E8', marker='o', zorder=5)
plt.scatter(stop_epoch_pure, val_losses_pure[stop_epoch_pure], color='#1F77B4', marker='o', zorder=5)
#plt.yscale('log')  # Optional: Use logarithmic scale for better visualization of loss
plt.xlabel('Epoch', fontsize=30)
plt.xlim(0,140)
plt.ylabel('Loss', fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=26)
plt.tick_params(axis='both', which='minor', labelsize=24)
#plt.title('Training and Validation Loss')
plt.legend(fontsize=28)
#plt.grid(True)

# Save the plot in the same model directory
plot_filename = os.path.join('/remote/gpu01a/pietschke/EoRFlow/output/paper_plots', 'loss_plot.pdf')
plt.savefig(plot_filename)
print(f"Loss plot saved to: {plot_filename}")


