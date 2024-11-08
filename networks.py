import torch
import torch.nn as nn
from typing import Literal, Tuple
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

    def forward(self, inputs: torch.Tensor):
        outputs_encoder = self.encoder(inputs)
        outputs_encoder_reshaped = self.reshape_encoder(outputs_encoder)
        outputs_decoder = self.decoder(outputs_encoder_reshaped)
        outputs = self.reshape_decoder(outputs_decoder)

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

    def forward(self, inputs: torch.Tensor):
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

    def forward(self, inputs: torch.Tensor):
        encoded_features = self.encoder(inputs)
        flatten_features = self.out_flatten(encoded_features)
        codede_outputs = self.codebook(flatten_features)
        if not self.training:
            return codede_outputs
        outputs = self.out_activation(codede_outputs)

        return outputs
