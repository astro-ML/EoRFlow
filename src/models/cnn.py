import torch
import torch.nn as nn
import torch.nn.functional as F

# an optional 3D CNN class for EoRFlow with FiLM layers for redshift conditioning

# FiLM layers for redshift conditioning
class FiLMGenerator(nn.Module):
    def __init__(self, redshift_dim, num_features_list):
        super(FiLMGenerator, self).__init__()
        self.layers = nn.ModuleList()
        for num_features in num_features_list:
            # Each layer produces 2 * num_features outputs: gamma and beta for FiLM
            self.layers.append(nn.Linear(redshift_dim, 2 * num_features))
        
    def forward(self, redshifts):
        gammas_betas = []
        for layer in self.layers:
            gamma_beta = layer(redshifts)  
            gammas_betas.append(gamma_beta)
        return gammas_betas

# 3D network for EoRFlow (15 params)
class CNN3D_15(nn.Module):
    def __init__(self):
        super(CNN3D_15, self).__init__()
    
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)  

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(128)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # FiLM Generator
        self.num_features_list = [32, 64, 128]
        self.film_generator = FiLMGenerator(redshift_dim=15, num_features_list=self.num_features_list) 

        self.fc_input_dim = 128 * 1 * 1 * 1
        self.fc1 = nn.Linear(self.fc_input_dim + 15, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # adjust output size as needed

        # Dropout
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, redshifts):
        # Generate FiLM parameters
        gammas_betas = self.film_generator(redshifts)  #
     
        # Convolutional Block 1
        x = self.conv1(x)
        #x = self.bn1(x)
        gamma_beta = gammas_betas[0]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)  # Each has shape (batch_size, num_features)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Reshape to (batch_size, num_features, 1, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta  # FiLM modulation
        x = F.relu(x)
        x = self.pool(x)  
  
        # Convolutional Block 2
        x = self.conv2(x)
        #x = self.bn2(x)
        gamma_beta = gammas_betas[1]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta
        x = F.relu(x)
        x = self.pool(x)  
 
        # Convolutional Block 3
        x = self.conv3(x)
        #x = self.bn3(x)
        gamma_beta = gammas_betas[2]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta
        x = F.relu(x)
        x = self.pool(x)  
  
        # Flatten
        x = x.view(x.size(0), -1)  
  
        # Concatenate redshift information
        x = torch.cat((x, redshifts), dim=1)
      
        # Fully Connected Layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output summary for the flow
        return x


