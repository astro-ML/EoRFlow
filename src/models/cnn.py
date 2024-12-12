import torch
import torch.nn as nn
import torch.nn.functional as F

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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



# 3D network for EoRFlow
class CNN3D_film(nn.Module):
    def __init__(self):
        super(CNN3D_film, self).__init__()
    
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)  

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(128)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # FiLM Generator
        self.num_features_list = [32, 64, 128]
        self.film_generator = FiLMGenerator(redshift_dim=30, num_features_list=self.num_features_list)

        self.fc_input_dim = 128 * 3 * 1 * 1
        self.fc1 = nn.Linear(self.fc_input_dim + 30, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # 10 , 20, 30

        # Dropout
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, redshifts):
        # Generate FiLM parameters
        gammas_betas = self.film_generator(redshifts)  #

        # Convolutional Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        gamma_beta = gammas_betas[0]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)  # Each has shape (batch_size, num_features)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Reshape to (batch_size, num_features, 1, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta  # FiLM modulation
        x = F.relu(x)
        x = self.pool(x)  

        # Convolutional Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        gamma_beta = gammas_betas[1]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta
        x = F.relu(x)
        x = self.pool(x)  

        # Convolutional Block 3
        x = self.conv3(x)
        x = self.bn3(x)
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




#__________________________________________________
# 2D network for EoRFlow
class CNN2D_film(nn.Module):
    def __init__(self):
        super(CNN2D_film, self).__init__()
        # Convolutional Blocks
        self.conv1 = nn.Conv2d(in_channels=30, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # FiLM Generator
        self.num_features_list = [64, 128, 256]
        self.film_generator = FiLMGenerator(redshift_dim=30, num_features_list=self.num_features_list)

        
        self.fc1 = nn.Linear(256 * 2 * 2 + 30, 256)  
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # Final output size

        # Dropout
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, redshifts):
        # Generate FiLM parameters
        gammas_betas = self.film_generator(redshifts)  

        # Convolutional Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        gamma_beta = gammas_betas[0]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)  
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta  # FiLM modulation
        x = F.relu(x)
        x = self.pool(x)  

        # Convolutional Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        gamma_beta = gammas_betas[1]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta
        x = F.relu(x)
        x = self.pool(x)  

        # Convolutional Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        gamma_beta = gammas_betas[2]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta
        x = F.relu(x)
        
        # Flatten
        x = x.view(-1, 256 * 2 * 2)

        # Concatenate redshift information
        x = torch.cat((x, redshifts), dim=1)  

        # Fully Connected Layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output summary for the flow
        return x

# 2D network for EoRFlow pretrained
class CNN2D_pretrained(nn.Module):
    def __init__(self):
        super(CNN2D_pretrained, self).__init__()
        # First Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=30, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # Second Convolutional Block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # Third Convolutional Block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        # Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 2 * 2 + 30, 256)  # Added 30 redshifts
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 30)  # Final output size (number of xH values)
        # Dropout
        self.dropout = nn.Dropout(p=0.3)
        # FiLM Generator
        self.film_generator = FiLMGenerator(redshift_dim=30, num_features_list=[64, 128, 256])

    def forward(self, x, redshifts):
        # Generate FiLM parameters
        gammas_betas = self.film_generator(redshifts)  # List of tensors for each conv layer

        # Convolutional Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        # Apply FiLM modulation
        gamma_beta = gammas_betas[0]  
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta
        x = F.relu(x)
        x = self.pool(x)

        # Convolutional Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        # Apply FiLM modulation
        gamma_beta = gammas_betas[1]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta
        x = F.relu(x)
        x = self.pool(x)

        # Convolutional Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        # Apply FiLM modulation
        gamma_beta = gammas_betas[2]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta
        x = F.relu(x)

        # Flatten
        x = x.view(-1, 256 * 2 * 2)

        # Concatenate redshifts
        x = torch.cat((x, redshifts), dim=1)  # Combine features with redshift information

        # Fully Connected Layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output for the flow model
        return x

# 3D network for EoRFlow pretrained
class CNN3D_pretrained(nn.Module):
    def __init__(self):
        super(CNN3D_pretrained, self).__init__()
        # Convolutional Blocks
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)  

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(128)

        # Pooling Layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # FiLM Generator
        self.num_features_list = [32, 64, 128]
        self.film_generator = FiLMGenerator(redshift_dim=30, num_features_list=self.num_features_list)

        self.fc_input_dim = 128 * 3 * 1 * 1
        self.fc1 = nn.Linear(self.fc_input_dim + 30, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 30)  # 10 , 20, 30

        # Dropout
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, redshifts):
        # Generate FiLM parameters
        gammas_betas = self.film_generator(redshifts)  

        # Convolutional Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        gamma_beta = gammas_betas[0]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)  
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta  # FiLM modulation
        x = F.relu(x)
        x = self.pool(x)

        # Convolutional Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        gamma_beta = gammas_betas[1]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta
        x = F.relu(x)
        x = self.pool(x)  

        # Convolutional Block 3
        x = self.conv3(x)
        x = self.bn3(x)
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






#______________________________________________
# for SKA challenge
class CNN2D_SKA(nn.Module):
    def __init__(self):
        super(CNN2D_SKA, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(32)

        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.film_generator = FiLMGenerator(redshift_dim=3, num_features_list=[16, 32])

        self.fc1 = nn.Linear(32 + 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  

        # Dropout
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, redshifts):
        # Generate FiLM parameters
        gammas_betas = self.film_generator(redshifts)  

        # Convolutional Block 1
        x = self.conv1(x)     
        x = self.bn1(x)
        
        gamma_beta = gammas_betas[0]  
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)  
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  
        beta = beta.unsqueeze(-1).unsqueeze(-1)    
        x = gamma * x + beta
        x = F.relu(x)
        x = self.pool(x)      

        # Convolutional Block 2
        x = self.conv2(x)     
        x = self.bn2(x)
        
        gamma_beta = gammas_betas[1]  
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1) 
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)        
        beta = beta.unsqueeze(-1).unsqueeze(-1)          
        x = gamma * x + beta
        x = F.relu(x)
        x = self.pool(x)      

        # Flatten
        x = x.view(x.size(0), -1)  

        x = torch.cat((x, redshifts), dim=1)

        # Fully Connected Layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  

        return x


class CNN3D_SKA(nn.Module):
    def __init__(self):
        super(CNN3D_SKA, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(16)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)

        self.film_generator = FiLMGenerator(redshift_dim=3, num_features_list=[16, 32])

        self.pool1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

        self.fc1 = nn.Linear(128 + 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, redshifts):
   
        gammas_betas = self.film_generator(redshifts)  

        # Convolutional Block 1
        x = self.conv1(x)  
        x = self.bn1(x)
 
        gamma_beta = gammas_betas[0]  
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1) 
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta
        x = F.relu(x)
        x = self.pool1(x)  

        # Convolutional Block 2
        x = self.conv2(x)  
        x = self.bn2(x)

        gamma_beta = gammas_betas[1]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1) 
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta
        x = F.relu(x)
        x = self.pool2(x)  

        # Flatten
        x = x.view(x.size(0), -1) 

        # Concatenate redshifts (3)
        x = torch.cat((x, redshifts), dim=1) 

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 

        return x






#_____________________
class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(32 * 1 * 1 + 3, 128)  # add 3 redshift dimensions
        self.fc2 = nn.Linear(128, 10)  # Final summary size to pass to the flow

    def forward(self, x, redshifts):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x))) 
        x = x.view(-1, 32 * 1 * 1)  # Flatten
        
        x = torch.cat((x, redshifts), dim=1)  # Concatenate along the feature dimension

        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)  # Output summary for the flow
        return x


class CNN2D_big(nn.Module):
    def __init__(self):
        super(CNN2D_big, self).__init__()
        # First Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=30, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # Batch Normalization
        # Second Convolutional Block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # Third Convolutional Block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        # Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 2 * 2 + 30, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # Final output size
        # Dropout
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, redshifts):
        # Convolutional Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  
        # Convolutional Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  
        # Convolutional Block 3
        x = F.relu(self.bn3(self.conv3(x)))            
        # Flatten
        x = x.view(-1, 256 * 2 * 2)

        x = torch.cat((x, redshifts), dim=1)  # Concatenate along the feature dimension

        # Fully Connected Layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output summary for the flow
        return x