import numpy as np
import matplotlib.pyplot as plt
import os

# Define the directory where the model files are stored
model_dir = './log_zscore_sigmoid'  # Update this to your actual directory

# Load the training and validation losses
train_losses = np.load(os.path.join(model_dir, 'train_losses.npy'))
val_losses = np.load(os.path.join(model_dir, 'val_losses.npy'))

# Plot the losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
#plt.yscale('log')  # Optional: Use logarithmic scale for better visualization of loss
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Save the plot in the same model directory
plot_filename = os.path.join(model_dir, 'loss_plot.pdf')
plt.savefig(plot_filename)
print(f"Loss plot saved to: {plot_filename}")


