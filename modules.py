import torch.nn as nn
import torch
from typing import Literal, List


class MLPBlock(nn.Module):
    def __init__(
        self, 
        dims: list, 
        activation: nn.Module | None = nn.ReLU(), 
        out_activation: nn.Module | None = None,
        dropout: float | None = 0.5
    ):
        """
        Initialize the module.

        Args:
            dim (list): List of dims in the mlp module.
            activation (Module): Activation after each linear layer other than the last one.
            out_activation (Module): Activation after the last linear layer.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.mlp = nn.Sequential()
        for in_dim, out_dim in zip(dims, dims[1:-1]):
            self.mlp.append(nn.Linear(in_dim, out_dim))
            self.mlp.append(activation)
            self.mlp.append(nn.Dropout(dropout))

        self.mlp.append(nn.Linear(dims[-2],dims[-1]))
        if out_activation is not None:
            self.mlp.append(out_activation)

    def forward(self, x:torch.Tensor):
        return self.mlp(x)
    

class ConvSubBlock(nn.Module):
    """Multiple 2-d convolutional layers with or without pooling layer."""
    def __init__(
        self, 
        channels: list, 
        pooling_method: Literal['avg','max', None] | None = 'max',
        activation: nn.Module | None = None
    ):
        """
        Initialize the model.
        
        Args:
            channels: Channels for the convolutional layers. 
                Convolutional layers will be formed for each consecutive pair of channels.
                For example, if the arg is [64, 128, 128], the first layer will be a 2-d conv layer
                with (64,128) as (in_channels, out_channels), and the second layer will be a 2-d conv layer
                with (128,128) as (in_channels, out_channels).
            pooling_method: The pooling method. `'avg'` indicates average pooling, 
                `'max'` indicates max pooling, `None` indicates do not form any pooling layer.
                Default to `'max'`.
            activation: The activation layer. `None` indicates do not form any activation layer. 
                This will be the last layer (if `None` is not passed). Default to `None`.
        """

        super().__init__()
        self.conv_layers = nn.Sequential()

        for in_channel, out_channel in zip(channels, channels[1:]):
            self.conv_layers.append(nn.Conv2d(in_channel, out_channel,kernel_size=3, padding=1))

        if pooling_method == 'avg':
            self.conv_layers.append(nn.AvgPool2d(2))
        elif pooling_method == 'max':
           self.conv_layers.append(nn.MaxPool2d(2))
        elif pooling_method is None:
            None
        else:
            raise ValueError(f'unknown pooling type {pooling_method}')
        
        if activation is not None:   
            self.conv_layers.append(activation)
        

    def forward(self, x:torch.Tensor):
        return self.conv_layers(x)
    

class ConvBlock(nn.Module):
    """The entire convolutional part."""
    def __init__(
        self, 
        channels_all: List[List[int]] | None = None,
        pooling_method: str | None = 'max',
        activation: nn.Module | None = None,
        models_args: List[dict] | None = None
    ):
        """
        Initialize the model.

        Args:
            channels_all (list): List where each element is also a list containing the channels for the sub convolutional block.
                If `None`, the channels must be passed through `models_args`. Default to `None`.
            pooling_method (str): The pooling method for all the pooling layers. Default to `'max'`.
            activation (Module): The activation function of the last layer of each sub convolutional blocks. 
                `None` indicates no activation function for the last layer. Default to `None`.
            models_args (list): List each element of which is the arg dict for the sub convolutional block. 
                If `None`, the channels must be passed through `channels_all`. Default to `None`.
        """
        
        assert (channels_all is None) ^ (models_args is None), 'channels_all and models_args cannot be both not None nor both None'
        
        super().__init__()
        self.conv_layers = nn.Sequential()

        if channels_all is not None:
            for channels in channels_all:
                self.conv_layers.append(ConvSubBlock(
                    channels=channels, 
                    pooling_method=pooling_method, 
                    activation=activation
                ))

        elif models_args is not None:
            for model_args in models_args:
                self.conv_layers.append(ConvSubBlock(**model_args))

    def forward(self, inputs):

        return self.conv_layers(inputs)


class ConvTransBlock(nn.Module):
    def __init__(
        self, 
        channels: list, 
        activation: nn.Module | None = nn.ReLU(),
        out_activation: nn.Module | None = nn.Sigmoid()
    ):
        """
        Initialize the model.

        Args:
            channels (list): List of channels of the module.
            activation (Module): Activation after each ConvTranspose2d other than the last one. Default to `nn.ReLU()`.
            out_activation (Module): Activation after the last ConvTranspose2d. Default to `nn.Sigmoid()`.
        """
        super().__init__()
        self.conv_trans_layers = nn.Sequential()
        for in_channels, out_channels in zip(channels,channels[1:-1]):
            self.conv_trans_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2,padding=1))
            if activation is not None:
                self.conv_trans_layers.append(activation)

        self.conv_trans_layers.append(nn.ConvTranspose2d(channels[-2], channels[-1], kernel_size=4, stride=2,padding=1))
        if out_activation is not None:
            self.conv_trans_layers.append(out_activation)

        
    def forward(self, x:torch.Tensor):
        return self.conv_trans_layers(x)
    

class CNN(nn.Module):
    """A simple CNN module."""
    def __init__(
        self, 
        channels_all: List[List[int]],
        mlp_dims: list,
        activations_conv: nn.Module | None = None,
        actications_mlp: nn.Module | None = nn.ReLU(),
        out_activation_mlp: nn.Module | None = None
    ):
        """
        Initialize the module.

        Args:
            channels_all (list): List where each element is also a list containing the channels for the sub convolutional block.
            mlp_dims (list): List of mlp dims.
            activations_conv (Module): Activation after each pooling layer in the convolutional part. Default to `None`.
            activations_mlp (Module): Activation after each linear layer in mlp other than the last one. Default to `nn.ReLU()`.
            out_activation_mlp (Module): Activation after the last linear layer in mlp. Default to `None`.
        """
        super().__init__()
        self.conv_block = ConvBlock(channels_all=channels_all, activation=activations_conv)
        self.flatten = nn.Flatten()
        self.mlp = MLPBlock(mlp_dims, activation=actications_mlp, out_activation=out_activation_mlp)

    def forward(self, inputs):
        conv_outputs = self.conv_block(inputs)
        flattened_outputs = self.flatten(conv_outputs)
        outputs = self.mlp(flattened_outputs)

        return outputs
    