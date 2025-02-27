import os
import numpy as np
import matplotlib.pyplot as plt

# Paths to your loss arrays
pure_dir = '/remote/gpu01a/pietschke/EoRFlow/output/full_EoR_pure_talk'
noise_dir = '/remote/gpu01a/pietschke/EoRFlow/output/full_EoR_noise_talk'

train_losses_pure = np.load(os.path.join(pure_dir, 'train_losses.npy'))
val_losses_pure = np.load(os.path.join(pure_dir, 'val_losses.npy'))

train_losses_noise = np.load(os.path.join(noise_dir, 'train_losses.npy'))
val_losses_noise = np.load(os.path.join(noise_dir, 'val_losses.npy'))

# Create a figure
plt.figure(figsize=(10,6))

# Plot noise model losses
plt.plot(train_losses_noise, label='Mock - Train', color='blue', linestyle='-')
plt.plot(val_losses_noise, label='Mock - Val', color='purple', linestyle='-')

# Plot pure model losses
plt.plot(train_losses_pure, label='Pure - Train', color='red', linestyle='-')
plt.plot(val_losses_pure, label='Pure - Val', color='darkorange', linestyle='-')



plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training and Validation Losses: Pure vs. Mock', fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the figure
output_path = '/remote/gpu01a/pietschke/EoRFlow/output/full_EoR_mutual'
os.makedirs(output_path, exist_ok=True)
plt.savefig(os.path.join(output_path, 'loss_comparison_pure_vs_noise.pdf'))
plt.close()

print("Loss comparison plot saved.")
