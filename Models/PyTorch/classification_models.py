import torch
import torch.nn as nn

# Simple CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, numChannels, numClasses):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.PReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )

        self.conv2 = nn.Sequential(         
            nn.Conv2d(in_channels=16, 
                      out_channels=32, 
                      kernel_size=5, 
                      stride=1, 
                      padding=2),     
            nn.PReLU(),                      
            nn.MaxPool2d(kernel_size=2),                
        )
        # fully connected layer, output 2 classes
        self.out = nn.Linear(32*50*25, numClasses) # -> PADDED
        # self.out = nn.Linear(32*32*32, numClasses) # -> SQUARE


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 50 * 25)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output  # Only return the logits
    
# NOT WORKING!

# Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self, base_cnn):
        super(SiameseNetwork, self).__init__()
        # Create two instances of the base CNN
        self.cnn1 = base_cnn
        self.cnn2 = base_cnn

    def forward_one(self, x):
        # Forward pass for one input
        output = self.cnn1(x)
        return output

    def forward(self, input1, input2):
        # Forward pass for both inputs
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2
    

# Contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - target) * torch.pow(euclidean_distance, 2) +
                                      (target) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive