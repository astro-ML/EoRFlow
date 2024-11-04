import torch
from torch.utils.data import DataLoader
from data_loader import PowerSpectrumDataset

# Assuming you've already defined PowerSpectrumDatasetFromNPZ
# Define the directory where your .npz files are stored
data_dir = './2D_data/test_10x10'  # Replace with the actual path to your .npz files

# Initialize the dataset and DataLoader
dataset = PowerSpectrumDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Using a small batch size for testing

# Test loading and iterating through the dataset
for i, (ps, labels) in enumerate(dataloader):
    print(f"Batch {i+1}")
    print("Power Spectra (flattened):")
    print(ps)  # Print the flattened power spectra
    print("Power Spectra Shape:", ps.shape)  # Should be (batch_size, 300)
    
    print("Labels (xH1, xH2, xH3):")
    print(labels)  # Print the target labels
    print("Labels Shape:", labels.shape)  # Should be (batch_size, 3)

    # Break after the first batch to avoid long output
    if i == 0:
        break
