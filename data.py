import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import correlation
from typing import Literal, Dict, List
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
        window (int): size of the sliding widow, i.e., width of the tranformed images.
        slide_step (int): sliding step of the window.
        apply_norm (bool): if apply nomalization to the images. Default to True.
        method (bool): the method to tranform time series into images. Must be `'gasf'`, `'gadf'`, or `'mtf'`. Default to `'gasf'`.

    Returns: 
        timeseries_images (ndarray): the transformed images with shape (n, 1, number of assets * window_length, window_length).
        dates (DataFrame): the corresonding dates with length (n,) and columns (`'DATE'`).
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
        window (int): size of the sliding widow, i.e., width of the tranformed images.
        slide_step (int): sliding step of the window.
        methods (str | list): list of methods to tranform 

    Returns:
        correlation_matrices (list): list with length of `len(df) - window_size + 1`, where each element list has length of `len(methods)` and contains different types of correlation matrices dertermined by `methods`.
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
        data (Tensor): the data with length n.
        dates (DataFrame): the dates with length n - lag.
        lag (int): lag for selecting taget images.
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
            data (ndarray): array with shape (n, 1, num_indices * window_size, window_size).
            dates (DataFrame): the corresponding datas to `data`.
            lag (int): lag for selecting the target images.
        """
        super().__init__()
        self.data = torch.tensor(data)
        self.lag = lag
        self.dates = dates[:-self.lag]

    def __len__(self):
        return len(self.data) - self.lag
    
    def __getitem__(self, index):

        return self.data[index], self.data[index + self.lag]
    

class TimeSeriesCorrelationsDatset:
    def __init__(
            self,
            data: List[List[pd.DataFrame]],
            dates: pd.DataFrame | None = None,
    ):  
        assert len(data) == len(dates)
        self.data = data
        self.dates = dates

    def __len__(self):
        return len(self.data)


def apply_pca(similarity:np.ndarray, variance_threshold=0.90):

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(similarity)

    pca = PCA(n_components=variance_threshold,svd_solver='full')
    pca_transformed = pca.fit_transform(scaled_data)

    return pca_transformed


if __name__ == '__main__':
    a = pd.DataFrame(
        {
            'a':[1,2,3,4,5,6,7,8,9],
            'b':[6,7,8,9,10,11,12,13,14]
        })
    #     , columns=['asset1','asset2','asset3','asset4','asset5']
    # )

    # test_rooling = [window.to_numpy() for window in a.rolling(2,min_periods=2)]
    # test_rooling = test_rooling[1:]

    # test = np.stack(test_rooling,axis=0)


    test = timeseries_to_image(a,window=3, method='gaf')
    print(test.dtype)

    plt.imshow(test[0],cmap='gray')
    plt.show()


    # print(a)
    # print(test)

    # print(test[:,:,0])
    # test = np.concatenate()

    # print(a)
    # print(a.to_numpy())
