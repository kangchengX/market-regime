import torch
import numpy as np
import pandas as pd
import correlation
from typing import Literal, List
from pyts.image import GramianAngularField, MarkovTransitionField
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def timeseries_to_image(
    timeseries: pd.DataFrame, 
    window: int, 
    slide_step: int,
    apply_norm: bool | None = True,
    method: Literal['gasf', 'gadf', 'mtf'] | None = 'gasf'
):
    """
    Transform multivariate timeseries into images. Within each window, an image will be formed by stacking the images for each time series transformed by `method`.
    
    Args:
        timeseries (DataFrame): DataFrame only with columns (`'DATE'`, index1, index2, ....) where indexn is the name of the index.
        window (int): Length of the sliding widow, i.e., width of the tranformed images.
        slide_step (int): Length of the sliding step .
        apply_norm (bool): If to apply nomalization to the images. Default to `True`.
        method (bool): the method to tranform time series into images. Must be `'gasf'`, `'gadf'`, or `'mtf'`. Default to `'gasf'`.

    Returns: 
        timeseries_images (ndarray): The transformed images with shape (n, 1, number of assets * window_length, window_length).
        dates (DataFrame): The corresonding dates with length (n,) and columns (`'DATE'`).
    """
    if method == 'gasf':
        transform = GramianAngularField()
    elif method == 'gadf':
        transform = GramianAngularField(method='difference')
    elif method == 'mtf':
        transform = MarkovTransitionField()
    else:
        raise ValueError(f'unknow value for arg method : {method}')
    
    dates = timeseries['DATE']
    timeseries = timeseries.drop(columns='DATE')

    # convert the DataFrame to an array with shape (number of dates - window_length + 1, window_length, number of assets)
    timeseries_windows = [timeseries_window.to_numpy() for timeseries_window in timeseries.rolling(window=window, step=slide_step)]
    timeseries_windows = timeseries_windows[int(np.ceil((window-1)/slide_step)):]
    timeseries_windows = np.stack(timeseries_windows) # number of windows * window_length * number of assets

    # get the corresponding dates
    dates = [dates_window.iloc[-1] for dates_window in dates.rolling(window=window, step=slide_step)]
    dates = dates[int(np.ceil((window-1)/slide_step)):]
    dates = pd.DataFrame({'DATE': dates})

    # tranform sereis into images
    # get a list of number of assets elements, each of which has shape (number of dates - window_length + 1, window_length, window_length)
    timeseries_images = [transform.transform(timeseries_windows[:,:,i]) for i in range(timeseries_windows.shape[-1])]

    # concatenate images along the row axis
    # the results array has shape (number of dates - window_length + 1, number of assets * window_length, window_length)
    timeseries_images = np.concatenate(timeseries_images, axis=1).astype(np.float32)

    # normalize the images
    if apply_norm:
        timeseries_images = (timeseries_images - np.mean(timeseries_images)) / (np.std(timeseries_images) + 1e-8)

    timeseries_images = np.expand_dims(timeseries_images,1)

    return timeseries_images, dates

def timeseries_to_correlation(
    timeseries: pd.DataFrame, 
    window: int, 
    slide_step: int,
    methods: Literal['pearson', 'kendall', 'spearman'] | list | None = ['pearson', 'kendall', 'spearman']
):
    """
    Transform multivariate time series into correlation matrices. Within each sliding window, a list with length of `len(methods)` containing different correlations will be gererated according to `methods`.

    Args:
        timeseries (DataFrame): DataFrame only with columns (`'DATE'`, index1, index2, ....) where indexn is the name of the index.
        window (int): Length of the sliding widow, i.e., width of the tranformed images.
        slide_step (int): Length of the sliding step.
        methods (str | list): List of types of correlations.

    Returns:
        correlation_matrices (list): List with length of `len(df) - window_size + 1`, where each element list has length of `len(methods)` and contains different types of correlation matrices dertermined by `methods`.
        dates (DataFrame): DataFrame with columns (`'DATE'`,)
    """
    dates = timeseries['DATE']
    timeseries = timeseries.drop(columns='DATE')

    correlation_matrices = correlation.calculate_rolling_correlation(
        df = timeseries, 
        window_size=window, 
        sliding_step=slide_step,
        methods=methods
    )

    dates = [dates_window.iloc[-1] for dates_window in dates.rolling(window=window, step=slide_step)]
    dates = dates[int(np.ceil((window-1)/slide_step)):]
    dates = pd.DataFrame({'DATE': dates})

    return correlation_matrices, dates


class TimeSeriesImagesDataset(Dataset):
    """
    The dataset for the timeseries data for the pre-text tasks. The input the current image, and the target is the image after `lag`.

    Attributes:
        data (Tensor): The data with length n.
        dates (DataFrame): The dates with length n - lag.
        lag (int): Lag for selecting taget images.
    """
    def __init__(
        self, 
        data: np.ndarray, 
        dates: pd.DataFrame | None = None,
        lag: int | None = 1
    ):
        """
        Initialize the dataset.
        
        Args:
            data (ndarray): Array with shape (n, 1, num_indices * window_size, window_size).
            dates (DataFrame): The corresponding datas to `data`.
            lag (int): Lag for selecting the target images. Note: the value of `lag` is for the selected images. In other words, 
                `lag` = 1 means the image for the timestamp t+sliding_step will be used as the target for the image for the timestamp t.
        """
        super().__init__()
        self.data = torch.tensor(data)
        self.lag = lag
        self.dates = dates[:-self.lag]

    def __len__(self):
        return len(self.data) - self.lag
    
    def __getitem__(self, index):

        return self.data[index], self.data[index + self.lag]
    

class TimeSeriesCorrelationsDataset:
    """
    The dataset for the correlation matrices.
    """
    def __init__(
        self,
        data: List[List[pd.DataFrame]],
        dates: pd.DataFrame | None = None
    ):  
        """
        Initialize the class.

        Args:
            data (list): A list each element of which is also a list containing different types of correlation matrices for the timestamp t.
            dates (DataFrame): A DataFrame only with the column (`'DATE'`,), and has the same length of `data`. Default to `None`.
        """
        assert len(data) == len(dates)
        self.data = data
        self.dates = dates

    def __len__(self):
        return len(self.data)


def apply_pca(similarity: np.ndarray, variance_threshold: int | float | None = 0.90):
    """
    Apply PCA on the similarity matrix.

    Args:
        similarity (ndarray): The similarity matrix.
        variance_threshold (int | float): If int, `variance_threshold` components will be kept. 
            If float, the components will be kept such that the total varient is over 90%.
        
    Returns:
        pca_transformed (ndarray): The resulted array, i.e., the features with row as a sample.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(similarity)

    pca = PCA(n_components=variance_threshold,svd_solver='full')
    pca_transformed = pca.fit_transform(scaled_data)

    return pca_transformed
