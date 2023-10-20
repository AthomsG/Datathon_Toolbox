import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time

class CNN(nn.Module):
    ''' 
    Convolutional neural network

    Parameters:

        @input_size: tuple of numbers (n_rows, n_cols)
        @num_classes: integer, number of classes
        @activation: string (relu, sigmoid, tanh, leakyrelu), activation function to use
        @prob: boolean, if it is a classification problem; it adds a softmax layer at the end

    Methods:

        @forward: forward pass of the model
        @train: train the model
        @regularization: calculate regularization component. Used in @train method

    ###########################################################################
    '''
    
    def __init__(self, input_size, num_classes, channels = 1, activation = 'relu'):
        
        super(CNN, self).__init__()
        
        self.input_size = input_size
        self.channels = channels
        
        # Define activation functions
        if activation == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation == 'relu':
            activation = nn.ReLU()
        elif activation == 'tanh':
            activation = nn.Tanh()
        elif activation == 'leakyrelu':
            activation = nn.LeakyReLU()

        last_channels = 16  
        out_features = (input_size[0] // 2 ) // 2 * (input_size[1] // 2 ) // 2 * last_channels
        

        # Define layers sequence
        layers = [nn.Conv2d(in_channels = channels, out_channels = 8, kernel_size = 5, padding = 2),
                  activation,
                  nn.MaxPool2d(kernel_size = 2),
                  \
                  nn.Conv2d(in_channels = 8, out_channels = last_channels, kernel_size = 5, padding = 2),
                  activation,
                  nn.MaxPool2d(kernel_size = 2),
                  \
                  nn.Flatten(),
                  nn.Linear(out_features, num_classes)]

        # Create model
        self.model = nn.Sequential(*layers)

    # --------------------------- FORWARD ---------------------------------------------

    def forward(self, x):
        '''
        Forward pass of the model
        '''
        return self.model(x)
    
    #------------------------------- TRAIN --------------------------------------------
    
    def train(self, data, epochs = 10, batch_size = 10, criterion = 'mse', optimizer = 'adam', lr = 0.01, reg = None, reg_weight = 0.1, verbose = 0, output = False, validation = 0.2):
        '''
        Train the model
        
        Parameters:
            
            @data: TensorDataset, tensor dataset of pairs (X, y), dataset to train the model. 
                    MAKE SURE THEY ARE FLOAT DTYPE and also type (X.len, n_features) and (y.len, 1)
            @epochs: integer, number of epochs
            @batch_size: integer, batch size
            @criterion: string (mse, cross_entropy), loss function
            @optimizer: string (adam, sgd), optimizer
            @lr: float, learning rate
            @reg: string (l1, l2), regularization
            @reg_weight: float, regularization weight
            @verbose: integer, print loss every verbose epochs. If 0, no printing
            @output: boolean, if True, return losses over epochs
            @validation: float, validation set size

        Output:
            
                @losses: pair of list of floats, training and validation losses over epochs
        '''

        data, val_data = random_split(data, [int((1 - validation) * len(data)), int(validation * len(data))])
        
        # Optimizer
        if optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr = lr)
        elif optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr = lr)

        # Criterion
        if criterion == 'mse':
            criterion = nn.MSELoss()
        elif criterion == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()

        # Regularization
        if reg == 'l1':
            p = 1
        elif reg == 'l2':
            p = 2

        # Create data loader
        data_loader = DataLoader(data, batch_size = batch_size, shuffle = True)

        # Store losses over epochs
        losses = []
        val_losses = []

        # Start timer
        start_time = time.time()

        # Range over epochs
        for i in range(1, epochs + 1):
            # Range over batches
            for x, y in data_loader:
                # Restart gradients
                optimizer.zero_grad()
                # Forward pass
                y_pred = self.forward(x)
                # Loss
                loss = criterion(y_pred, y)
                # Regularization
                if reg:
                    loss += self.regularization(p, reg_weight)
                # Backward pass
                loss.backward()
                optimizer.step()

            # Validation loss
            with torch.no_grad():
                val_loss = 0
                for val_x, val_y in val_data:
                    val_x = val_x.view(-1, self.channels, self.input_size[0], self.input_size[0])
                    val_y_pred = self.forward(val_x).view(-1)
                    val_loss += criterion(val_y_pred, val_y)
                if reg:
                    val_loss += self.regularization(p, reg_weight)
                val_losses.append(val_loss.item() / len(val_data))

            # Store loss
            losses.append(loss.item())

            # Print loss
            if verbose and i % verbose == 0:
                    time_passed = round(time.time() - start_time, 2)
                    print('Epoch {} | Train Loss: {} | Validation Loss: {} | Time: {}s'.format(i, losses[-1], val_losses[-1], time_passed), end = '\n')
                    start_time = time.time()
            
        
        # Return losses
        if output:
            return losses, val_losses


    # --------------------------- CALCULATE REG COMPONENT ---------------------------------------------
    def regularization(self, p, weight):
        '''
        Calculate regularization component

        Parameters:

            @p: integer, p-norm

        Output:

            @reg: torch float, regularization component
        '''
        reg = 0
        for param in self.parameters():
            reg += torch.norm(param, p = p)
        return weight * reg