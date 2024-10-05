import torch
import torch.nn as nn
from typing import List

class CNN_LSTM(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_layers = config.num_layers 
        self.hidden_dim = config.hidden_dim
        self.conv1d_layer = nn.Conv1d(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            kernel_size=config.kernel_size,
            padding="same"
        )
        self.tanh = nn.Tanh()
        self.max_pool1d = nn.MaxPool1d(kernel_size=config.pool_kernel_size)
        self.relu = nn.ReLU()
        self.lstm_layer = nn.LSTM(input_size=config.out_channels,
                                  hidden_size=config.hidden_dim,
                                  num_layers=config.num_layers)
        self.fc_last = nn.Linear(config.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def get_hidden(self, num_seq):
        h0 = torch.randn(self.num_layers, num_seq, self.hidden_dim)
        c0 = torch.randn(self.num_layers, num_seq, self.hidden_dim)
        return (h0, c0)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x =  self.tanh(self.conv1d_layer(x))
        x = self.relu(self.max_pool1d(x))
        x = x.permute(0, 2, 1)
        x, (h0, c0) = self.lstm_layer(x, self.get_hidden(x.size(1)))
        x = self.tanh(x)
        x = self.sigmoid(self.fc_last(x[:, -1, :]))
        return x
    

# model for simple mlp training
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, config) -> None:
        '''
        Initialize the SimpleNeuralNetwork class with specified parameters.

        Parameters:
        input_dim (int): Number of input features.
        hidden_dim (List[int]): List of hidden layer dimensions.
        output_dim (int): Output dimension.
        dropout (float): Dropout probability.
        activation (str): Type of activation function to use ('relu', 'tanh', 'leaky_relu').
        '''
        super(SimpleNeuralNetwork, self).__init__()

        self.config = config
        self.activation_fn = self._get_activation_function(config.activation)
        
        # Creating layers dynamically
        layers = []
        input_dim = config.input_dim

        for h_dim in config.hidden_dim:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))  # BatchNorm after linear
            layers.append(self.activation_fn)  # Activation
            layers.append(nn.Dropout(config.dropout))  # Dropout
            input_dim = h_dim

        # Output layer (without batch norm and dropout)
        layers.append(nn.Linear(input_dim, config.output_dim))
        layers.append(nn.Sigmoid())
        # Combine all layers into a sequential module
        self.network = nn.Sequential(*layers)
        
        # Optional: Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        ''' Custom weight initialization '''
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _get_activation_function(self, activation: str):
        ''' Helper function to get activation function '''
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        ''' Forward pass '''
        return self.network(x)