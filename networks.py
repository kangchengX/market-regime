import torch
import torch.nn as nn
from typing import Literal, List, Tuple
from constant import TORCH_MODULE_MAP, CUSTOM_MODULE_MAP


class AutoEncoder(nn.Module):
    """Implementation of the autoencoder."""
    def __init__(
        self, 
        encoder: Literal['cnn', 'mlp', 'conv', 'conv_trans'], 
        encoder_args: dict,
        decoder: nn.Module,
        decoder_args: dict,
        reshape_encoder: Literal['flatten', 'unflatten', None] | None = None,
        reshape_decoder: Literal['flatten', 'unflatten', None] | None = None,
        inter_feature_shape: Tuple[int, int, int] | None = None,
        out_feature_shape: Tuple[int, int, int] | None = None
    ):
        """
        Initialize the model.
        
        Args:
            encoder (str): The encoder module.
            encoder_args (dict): Args of the encoder module.
            decoder (str): The decoder module.
            decoder_args (dict): Args of the decoder module.
            reshape_encoder (str | None): If 'flatten', flatten the outputs of the encoder; if 'unflatten', unflatten the outputs of the encoder to [batch, channels, height, width]
                according to `inter_feature_shape`; if `None`, not reshape. Default to `None`.
            reshape_decoder (str | None): If 'flatten', flatten the outputs of the decoder; if 'unflatten', unflatten the outputs of the decoder to [batch, channels, height, width]
                according to `out_feature_shape`; if `None`, not reshape. Default to `None`.
            inter_feature_shape (tuple): (channels, height, width) if the input of the decoder is an image. Default to `None`.
            out_feature_shape (tuple): (channels, height, width) if the final output of the model is an image. Default to `None`.
        """
        
        super().__init__()
        self.encoder = CUSTOM_MODULE_MAP[encoder](**encoder_args)
        self.decoder = CUSTOM_MODULE_MAP[decoder](**decoder_args)

        if reshape_encoder == 'flatten':
            self.reshape_encoder = nn.Flatten()
        elif reshape_encoder == 'unflatten':
            self.reshape_encoder = nn.Unflatten(1,inter_feature_shape)
        elif reshape_encoder is None:
            self.reshape_encoder = nn.Identity()
        else:
            raise ValueError(f'Unsupported reshape_encoder {reshape_encoder}')

        if reshape_decoder == 'flatten':
            self.reshape_decoder = nn.Flatten()
        elif reshape_decoder == 'unflatten':
            self.reshape_decoder = nn.Unflatten(1,out_feature_shape)
        elif reshape_decoder is None:
            self.reshape_decoder = nn.Identity()
        else:
            raise ValueError(f'Unsupported reshape_decoder {reshape_decoder}')

    def forward(self, inputs):
        outputs_encoder = self.encoder(inputs)
        outputs_encoder_reshaped = self.reshape_encoder(outputs_encoder)
        outputs_decoder = self.decoder(outputs_encoder_reshaped)
        outputs = self.reshape_decoder(outputs_decoder)

        return outputs
    

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


class CodeBook(nn.Module):
    """The CodeBook, which is an interpretation of the end-to-end model from the perspective of a two-stage model."""
    def __init__(self, num_feature: int, num_codes: int):
        """
        Initialize the module. This is a dense layer.

        Args:
            num_feature (int): The 'feature' dimension, i.e., input dimension of the dense layer.
            num_codes (int): Number of the codes in the CodeBook, i.e., the given number of regimes, or, the output dimension of this dense layer.
        """
        super().__init__()
        self.num_codes = num_codes
        self.sim = nn.Linear(in_features=num_feature, out_features=self.num_codes)

    def forward(self, inputs:torch.Tensor):
        simlarity = self.sim(inputs)

        if self.training:
            return simlarity
        
        indices = simlarity.argmax(dim=-1)

        return indices
    

class AutoEncoderCodeBook(nn.Module):
    """Implementation of the autoencoder with CodeBook, i.e., end to end model for regime identification."""
    def __init__(
        self, 
        encoder: Literal['cnn', 'mlp', 'conv', 'conv_trans'], 
        encoder_args: dict,
        num_features: int,
        num_codes: int,
        out_flatten: bool | None = False,
        out_activation: str | nn.Module | None = None
    ):
        """Initialize the module.
        
        Args:
            encoder (str): The `encoder` module.
            encoder_args (dict): Args of the `encoder` module.
            num_feature (int): The 'feature' dimension, i.e., input dimension of the last dense layer of the whole model.
            num_codes (int): Number of the codes in the CodeBook, i.e., the given number of regimes, or, the output dimension of this last layer of the whole model.
            out_flatten (bool): If `True`, the 'features' will be flattened before being fed into the CodeBook, i.e., the last layer of the whole model.
                Default to `False`.
            out_activation (str|Module|None): If not `None`, the 'features' will go through the given activation function before being fed into the CodeBook, 
                i.e., the last layer of the whole model. Default to `None`.
        """
        
        super().__init__()
        self.encoder = CUSTOM_MODULE_MAP[encoder](**encoder_args)
        self.codebook = CodeBook(num_feature=num_features, num_codes=num_codes)

        if out_flatten:
            self.out_flatten = nn.Flatten()
        else:
            self.out_flatten = nn.Identity()

        if out_activation is not None:
            if isinstance(out_activation, str):
                self.out_activation = TORCH_MODULE_MAP[out_activation]
            else:
                self.out_activation = out_activation
        else:
            self.out_activation = nn.Identity()

    def forward(self, inputs):
        encoded_features = self.encoder(inputs)
        flatten_features = self.out_flatten(encoded_features)
        codede_outputs = self.codebook(flatten_features)
        if not self.training:
            return codede_outputs
        outputs = self.out_activation(codede_outputs)

        return outputs
