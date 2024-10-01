from __future__ import annotations
import torch.nn as nn
import seaborn as sns
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from networks import CNN, MLPBlock, ConvBlock, ConvTransBlock

PYTHON_INTER = 'f:/venvs/pytorch/Scripts/python.exe'

SECURITY = {'MXWO Index', 'DLJHVAL Index', 'SPX Index'}

COLUMN_MAP_RETURN = {
    'SPX Index' : 'LOG',
    'MXWO Index' :  'LOG',
    'DXY Curncy': 'LOG',
    'XAU Curncy': 'LOG',
    'SPGSIN Index': 'LOG',
    'USGG10YR Index' : 'DIFF',
    'VIX Index' : 'DIFF',
    'DLJHSTW Index' : 'DIFF',
    'CSI BARC Index' : 'DIFF'
}

TORCH_MODULE_MAP = {
    'softmax' : nn.Softmax(dim=-1)
}

CUSTOM_PALETTE = {
    'with pca': sns.color_palette()[0],
    'without pca': sns.color_palette()[1],
    '10': sns.color_palette()[0],
    '1' : sns.color_palette()[1],
    10 : sns.color_palette()[0],
    1 : sns.color_palette()[1],
}

CUSTOM_MODULE_MAP = {
    'cnn' : CNN,
    'mlp' : MLPBlock,
    'conv' : ConvBlock,
    'conv_trans' : ConvTransBlock
}
