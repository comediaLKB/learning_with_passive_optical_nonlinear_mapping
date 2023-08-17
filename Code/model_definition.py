# Author: Fei Xia
# Create Date: July-20-2022
# Last Update: August-10-2023
# Part of the code was inpsired from repo by Martin Stein
# If you use any of the datasets or code in this repository for your research, please consider citing our work:
# Xia, F., Kim, K., Eliezer, Y., Shaughnessy, L., Gigan, S., & Cao, H. (2023). Deep Learning with Passive Optical Nonlinear Mapping. arXiv preprint arXiv:2307.08558.


import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=None, use_batchnorm=False, activation_function='relu', **kwargs):
        '''
        Configurable Deep Neural Network with fully connected layers and customizable
        nonlinear activation functions.

        Args:
            input_size (int): Size of input layer
            output_size (int): Size of output layer
            hidden_layers (list of int): Sizes of hidden layers
            use_batchnorm (bool): If True, uses batch normalization between each layer
            activation_function (str): Type of activation function: 'relu', 'tanh', or 'sigmoid'
        '''
        super(MLP, self).__init__()

        # Default hidden layer configuration if not provided
        if hidden_layers is None:
            hidden_layers = [100, 100]
        self.use_batchnorm = use_batchnorm

        # Setting the activation function
        activation_functions = {
            'relu': torch.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid
        }
        self.activation = activation_functions[activation_function]

        # Incorporate the input size into the layer configuration
        layer_sizes = [input_size] + hidden_layers

        # Define fully connected layers
        self.fc_layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)
        ])

        # Define the final output layer
        self.output_layer = nn.Linear(layer_sizes[-1], output_size)

        # If batch normalization is enabled, define batch normalization layers
        if use_batchnorm:
            self.batch_norm_layers = nn.ModuleList([
                nn.BatchNorm1d(size) for size in layer_sizes[1:]
            ])

    def forward(self, inputs):
        '''
        Forward pass through the network.

        Args:
            inputs (torch.Tensor): Input tensor with shape [batch_size, input_size]
        Returns:
            torch.Tensor: Output tensor from the network
        '''
        # Pass through each fully connected layer
        for idx, fc_layer in enumerate(self.fc_layers):
            inputs = fc_layer(inputs)
            if self.use_batchnorm:
                inputs = self.batch_norm_layers[idx](inputs)
            inputs = self.activation(inputs)

        # Return the final output
        return self.output_layer(inputs)
