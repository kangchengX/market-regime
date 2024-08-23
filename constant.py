import torch.nn as nn

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

STRING_MODULE_MAP = {
    'softmax' : nn.Softmax(dim=-1)
}