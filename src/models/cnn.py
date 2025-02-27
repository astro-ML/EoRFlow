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

# 3D network for EoRFlow 10 param 
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
        self.film_generator = FiLMGenerator(redshift_dim=15, num_features_list=self.num_features_list) #30

        self.fc_input_dim = 128 * 1 * 1 * 1
        self.fc1 = nn.Linear(self.fc_input_dim + 15, 256) # +30
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
        self.film_generator = FiLMGenerator(redshift_dim=30, num_features_list=self.num_features_list) #30

        self.fc_input_dim = 128 * 3 * 1 * 1
        self.fc1 = nn.Linear(self.fc_input_dim + 30, 256) # +30
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
class CNN2D_3param(nn.Module):
    def __init__(self):
        super(CNN2D_3param, self).__init__()
        # Convolutional Blocks
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # FiLM Generator
        self.num_features_list = [64, 128, 256]
        self.film_generator = FiLMGenerator(redshift_dim=3, num_features_list=self.num_features_list)

        
        self.fc1 = nn.Linear(256 * 2 * 2 + 3, 256)  
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
        self.fc3 = nn.Linear(64, 9)  

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


# larger CNN for SKA

class CNN3D_SKA_Larger(nn.Module):
    def __init__(self, in_ch: int = 1, ch: int = 16, N_parameter: int = 10, sigmoid: bool = False):
        super(CNN3D_SKA_Larger, self).__init__()
        
        self.conv1 = nn.Conv3d(in_ch, ch, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv2 = nn.Conv3d(ch, ch, kernel_size=(3,3,3), stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        self.conv3 = nn.Conv3d(ch, 2*ch, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv3_zero = nn.Conv3d(2*ch, 2*ch, kernel_size=(3,3,3), stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

        self.conv4 = nn.Conv3d(2*ch, 4*ch, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv4_zero = nn.Conv3d(4*ch, 4*ch, kernel_size=(3,3,3), stride=1, padding=1)

        self.avg = nn.AdaptiveAvgPool3d((1,1,1))
        self.flatten = nn.Flatten()

        self.film_generator = FiLMGenerator(redshift_dim=3, num_features_list=[ch, 2*ch])

        self.linear1 = nn.Linear(67, 128, bias=True)
        self.linear2 = nn.Linear(128, 128, bias=True)
        self.linear3 = nn.Linear(128, 128, bias=True)
        self.out = nn.Linear(128, N_parameter, bias=True)

        self.sigmoid = sigmoid

    def apply_film(self, x, gamma_beta):
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1) 
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta
        return x

    def forward(self, x: torch.Tensor, redshifts: torch.Tensor) -> torch.Tensor:
        # Generate FiLM parameters
        gammas_betas = self.film_generator(redshifts)  

        # Block 1
        x = F.relu(self.conv1(x))
        # FiLM after conv1
        x = self.apply_film(x, gammas_betas[0])
        x = F.relu(self.conv2(x))
        x = self.pool1(x)  

        # Block 2
        x = F.relu(self.conv3(x))
        # FiLM after conv3
        x = self.apply_film(x, gammas_betas[1])
        x = F.relu(self.conv3_zero(x))
        x = self.pool2(x) 

        # Block 3
        x = F.relu(self.conv4(x))       
        x = F.relu(self.conv4_zero(x))  

        x = self.avg(x)    
        x = self.flatten(x) 

        x = torch.cat((x, redshifts), dim=1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        if self.sigmoid:
            x = torch.sigmoid(self.out(x))
        else:
            x = self.out(x)

        return x


class CNN3D_PIE(nn.Module):
    def __init__(self, in_ch: int = 1, ch: int = 16, N_parameter: int = 10, sigmoid: bool = False):
        super(CNN3D_PIE, self).__init__()
        
        self.conv1 = nn.Conv3d(in_ch, ch, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv2 = nn.Conv3d(ch, ch, kernel_size=(3,3,3), stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        self.conv3 = nn.Conv3d(ch, 2*ch, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv3_zero = nn.Conv3d(2*ch, 2*ch, kernel_size=(3,3,3), stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

        self.conv4 = nn.Conv3d(2*ch, 4*ch, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv4_zero = nn.Conv3d(4*ch, 4*ch, kernel_size=(3,3,3), stride=1, padding=1)

        self.avg = nn.AdaptiveAvgPool3d((1,1,1))
        self.flatten = nn.Flatten()

        self.film_generator = FiLMGenerator(redshift_dim=30, num_features_list=[ch, 2*ch])

        self.linear1 = nn.Linear(64 + 30, 128, bias=True)
        self.linear2 = nn.Linear(128, 128, bias=True)
        self.linear3 = nn.Linear(128, 128, bias=True)
        self.out = nn.Linear(128, N_parameter, bias=True)

        self.sigmoid = sigmoid

    def apply_film(self, x, gamma_beta):
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1) 
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta
        return x

    def forward(self, x: torch.Tensor, redshifts: torch.Tensor) -> torch.Tensor:
        # Generate FiLM parameters
        gammas_betas = self.film_generator(redshifts)  

        # Block 1
        x = F.relu(self.conv1(x))
        # FiLM after conv1
        x = self.apply_film(x, gammas_betas[0])
        x = F.relu(self.conv2(x))
        x = self.pool1(x)  

        # Block 2
        x = F.relu(self.conv3(x))
        # FiLM after conv3
        x = self.apply_film(x, gammas_betas[1])
        x = F.relu(self.conv3_zero(x))
        x = self.pool2(x) 

        # Block 3
        x = F.relu(self.conv4(x))       
        x = F.relu(self.conv4_zero(x))  

        x = self.avg(x)    
        x = self.flatten(x) 

        x = torch.cat((x, redshifts), dim=1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        if self.sigmoid:
            x = torch.sigmoid(self.out(x))
        else:
            x = self.out(x)

        return x




class CNN3D_exp(nn.Module):
    def __init__(self):
        super(CNN3D_exp, self).__init__()
    
        # Block 1: Two convs, FiLM after first conv
        self.conv1a = nn.Conv3d(1, 32, 3, padding=1)
        self.bn1a = nn.BatchNorm3d(32)
        self.conv1b = nn.Conv3d(32, 32, 3, padding=1)
        self.bn1b = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d((2,2,2))

        # Block 2: Two convs, FiLM after first conv in block 2
        self.conv2a = nn.Conv3d(32, 64, 3, padding=1)
        self.bn2a = nn.BatchNorm3d(64)
        self.conv2b = nn.Conv3d(64, 64, 3, padding=1)
        self.bn2b = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d((2,2,2))

        # Block 3: One conv, FiLM after this conv
        self.conv3 = nn.Conv3d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d((2,2,2))

        # FiLM for conv1a, conv2a, conv3 only = 3 FiLM layers total
        self.film_generator = FiLMGenerator(redshift_dim=30, num_features_list=[32,64,128])

        # After block3: Adapt to final dims (depends on input size)
        # Suppose final spatial dims are (128,1,1,1) after all pooling
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128+30, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,30)
        self.dropout = nn.Dropout(0.3)

    def apply_film(self, x, gamma_beta):
        gamma, beta = torch.chunk(gamma_beta,2,dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gamma*x + beta

    def forward(self, x, redshifts):
        gammas_betas = self.film_generator(redshifts)

        # Block 1
        x = self.conv1a(x)
        x = self.bn1a(x)
        x = self.apply_film(x, gammas_betas[0])  # FiLM after conv1a
        x = F.relu(x)
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)

        # Block 2
        x = self.conv2a(x)
        x = self.bn2a(x)
        x = self.apply_film(x, gammas_betas[1])  # FiLM after conv2a
        x = F.relu(x)
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.apply_film(x, gammas_betas[2])  # FiLM after conv3
        x = F.relu(x)
        x = self.pool3(x)

        x = nn.AdaptiveAvgPool3d((1,1,1))(x)
        x = x.view(x.size(0), -1)  # (batch_size,128)
        x = torch.cat((x, redshifts), dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#__________________________________
# Optuna version
class CNN3D_tuned(nn.Module):
    """
    A 3D CNN that can have 1..4 convolutional blocks (each with FiLM),
    followed by exactly 4 linear layers. The last linear layer outputs
    final_out_dim.

    Convolution blocks:
        - conv + bn + FiLM + ReLU + pool
    We define 4 blocks but only run `num_conv_blocks` in forward.

    Linear layers (always 4):
        fc1( in_features -> 128 ), fc2(128->128), fc3(128->128), fc4(128->final_out_dim)
        with dropout after each of the first three FC layers (if desired).

    For dimension stability, we do an AdaptiveAvgPool3d to (1,1,1) after the
    last used conv block, so the flatten shape is always out_channels_of_that_block.
    Then we add `redshift_dim` to that shape as well, for the head.
    """

    def __init__(
        self,
        num_conv_blocks: int = 4,    # 1..4
        final_out_dim: int = 28,     # e.g. 3..50
        dropout: float = 0.3,
        redshift_dim: int = 30,
    ):
        super().__init__()
        if not (1 <= num_conv_blocks <= 4):
            raise ValueError("num_conv_blocks must be between 1 and 4.")
        if final_out_dim < 1:
            raise ValueError("final_out_dim must be >= 1.")

        self.num_conv_blocks = num_conv_blocks
        self.final_out_dim = final_out_dim
        self.dropout_p = dropout
        self.redshift_dim = redshift_dim

        # ----- 1) Define up to 4 conv blocks (Conv3d+BN+pool) -----
        # We'll pick out_channels for 4 blocks: [32, 64, 128, 256]
        # They are all defined, but only used up to `num_conv_blocks`.
        out_channels_list = [32, 64, 128, 256]

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d((2,2,2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d((2,2,2))

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d((2,2,2))

        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm3d(256)
        #self.pool4 = nn.MaxPool3d((2,2,2))

        # Which out_channels is the final block?
        final_c = out_channels_list[num_conv_blocks - 1]

        # ----- 2) FiLM generator: we create FiLM for the blocks we actually use. -----
        # e.g. if num_conv_blocks=2 => we only want gamma/beta for the first 2 conv blocks.
        used_out_channels = out_channels_list[:num_conv_blocks]
        self.film_generator = FiLMGenerator(redshift_dim, used_out_channels)

        # ----- 3) Adaptive pool so that after the last conv, shape is (batch_size, final_c, 1,1,1). -----
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1,1,1))

        # ----- 4) Fully-connected layers (always 4 layers):
        # fc1: (final_c + redshift_dim) -> 128
        # fc2: 128 -> 128
        # fc3: 128 -> 128
        # fc4: 128 -> final_out_dim
        self.fc1 = nn.Linear(final_c + redshift_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, final_out_dim)

        # We apply dropout after fc1, fc2, fc3 (optionally).
        self.dropout = nn.Dropout(p=self.dropout_p)


    def apply_film(self, x, gamma_beta):
        """
        x: shape (batch_size, out_ch, D, H, W)
        gamma_beta: shape (batch_size, 2*out_ch)
        """
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (batch_size, out_ch,1,1,1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gamma*x + beta


    def forward(self, x, redshifts):
        """
        x: shape (batch_size,1,D,H,W)   [the 3D volume, e.g. (30,10,10) after unsqueeze(1)]
        redshifts: shape (batch_size, redshift_dim)
        """
        # 1) Generate FiLM parameters for each conv block used
        gamma_betas = self.film_generator(redshifts)  
        # length = num_conv_blocks

        # 2) Pass through blocks:

        # block1
        if self.num_conv_blocks >= 1:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.apply_film(x, gamma_betas[0])  # FiLM
            x = F.relu(x)
            x = self.pool1(x)

        # block2
        if self.num_conv_blocks >= 2:
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.apply_film(x, gamma_betas[1])
            x = F.relu(x)
            x = self.pool2(x)

        # block3
        if self.num_conv_blocks >= 3:
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.apply_film(x, gamma_betas[2])
            x = F.relu(x)
            x = self.pool3(x)

        # block4
        if self.num_conv_blocks == 4:
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.apply_film(x, gamma_betas[3])
            x = F.relu(x)
            #x = self.pool4(x)

        # 3) Adaptive pool => (batch_size, final_c,1,1,1)
        x = self.adaptive_pool(x)
        # Flatten => (batch_size, final_c)
        x = x.view(x.size(0), -1)

        # 4) Concat redshifts => (batch_size, final_c + redshift_dim)
        x = torch.cat([x, redshifts], dim=1)

        # 5) Four linear layers always
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        # final out
        x = self.fc4(x)

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