import numpy as np
import pandas as pd
from typing import Union


def calculate_regime_durations(regimes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate regime durations.
    
    Args:
        regimes_df (DataFrame): DateFrame that includes columns (`'DATE'`, `'regime'`). Other columns will be ignored.

    Returns:
        regime_durations (DataFrame): DateFrame only with columns (`'regime'`, `'period'`, `'duration'`).
    """
    regimes_df = regimes_df[['DATE', 'regime']]
    regimes_df['regime_if_new'] = regimes_df['regime'] != regimes_df['regime'].shift(1)
    regimes_df['period'] = regimes_df['regime_if_new'].cumsum()

    regime_durations = regimes_df.groupby(by=['regime', 'period']).size().reset_index(name='duration')

    return regime_durations


def calculate_regime_durations_statistics(regime_durations: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mean, std, 25, 50, 70 quantiles for regime durations and the total days and the corresponding percentage for each regime and all regime periods.
    
    Args:
        regimes_durations (DataFrame): dataframe that includes columns (`'regime'`, `'duration'`). Other columns will be ignored.
            Note, the 'duration'` is the duration of the period when the regime keeps the same. Not the total duration of a regime.

    Returns:
        statistics (DataFrame): dataframe with columns (`'regime'`, `'mean'`, `'std'`, `'count'`, `'25 quantile'`, `'median'`, `'75 quantile'`, `'percent'`).
    """

    total_dates = regime_durations['duration'].sum()

    # for each regime
    statistics = {
        regime: 
        np.concatenate(
            (
                [dura['duration'].mean(), dura['duration'].std(), dura['duration'].sum()], 
                np.quantile(dura['duration'], [0.25,0.5,0.75])
            )
        ) 
        for regime, dura in regime_durations.groupby(by='regime')
    }

    # for the whole period
    statistics['all'] = np.concatenate(
        (
            [regime_durations['duration'].mean(), regime_durations['duration'].std(), regime_durations['duration'].sum()], 
            np.quantile(regime_durations['duration'], [0.25,0.5,0.75])
        )
    )

    # to DateFrame
    statistics = pd.DataFrame.from_dict(statistics, orient='index', columns=['mean','std','count', '25 quantile','median','75 quantile'])
    
    # add 'percent'
    statistics['percent'] = statistics['count'] / total_dates 
    statistics = statistics.reset_index(names='regime')

    return statistics


def calculate_return_metrics(
        df: pd.DataFrame, 
        freq_alias: str | None = None, 
        freq_length: int | None = 1
):
    """
    Calculate 5 metrics for returns - average return, std, annualized return, annualized std, sharpe ratio. Note, the return period used for
    the average return and std are givin by `freq_alias` and `freq_length` (and alias and length should be constant). For example,
    alias `'W'` and length 5 for average weekly return.

    Args:
        df (DataFrame): the data only with columns (`'DATE'`, ...) (the other columns are returns).
        freq_alias (str | None): alias for the length of the return period for average return and std. This can be `'W'`, `'2W'`, `'ME'`. 
            Default is `None` for 1 day. Full list see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
        freq_length (int): length of the return period (in days). Default to `1`.

    Returns:
        metrics (DataFrame): dataframe with indices as indexes and metrics as columns.
    """

    # group by frequency
    if freq_alias is not None:
        # not to use sum() here because there may be some missing data during the workdaying during the return period
        df_returns = df.groupby(pd.Grouper(key='DATE',freq=freq_alias),sort=False).mean()*freq_length
    else:
        df_returns = df.drop(columns='DATE')

    # calculate metrics
    average_return = df_returns.mean()
    std = df_returns.std()
    ann_average_return = (252/freq_length) * average_return
    ann_std = np.sqrt(252/freq_length) * std
    sharpe_ratio = ann_average_return / ann_std

    metrics = {
        'average_return' : average_return,
        'std' : std,
        'ann_return' : ann_average_return,
        'ann_std' : ann_std,
        'sharpe_ratio' : sharpe_ratio
    }

    return pd.DataFrame(metrics)


def calculate_return_metrics_within_regime(
        df: pd.DataFrame, 
        freq_alias: str | None = None, 
        freq_length: int | None = 1
) ->pd.DataFrame:
    """
    Calculate 5 metrics for returns - average return, std, annualized return, annualized std, sharpe ratio within each regime. 
    Note, the return period used for the average return and std are givin by `freq_alias` and `freq_length` (and alias and length should be constant). 
    For example, alias `'W'` and length 5 for average weekly return.
    
    Args:
        df: the data only with columns (`'DATE'`, `'regime'`, ...) (the other columns are returns).
        freq_alias (str | None): alias for the length of the return period for average return and std. This can be `'W'`, `'2W'`, `'ME'`. 
            Default is `None` for 1 day. Full list see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
        freq_length (int): length of the return period (in days). Default to `1`.

    Returns:
        results (dict): dict with regimes as keys as dataframe as values. The dataframe has indices as indexes and metrics as columns.
    """

    results = df.groupby('regime').apply(
        calculate_return_metrics,
        freq_alias = freq_alias,
        freq_length = freq_length,
        include_groups=False
    ).reset_index(names=['regime', 'index'])

    return results


def calculate_return_metrics_last_date_of_period(
        df: pd.DataFrame,
        freq_alias: str | None = None, 
        freq_length: int | None = 1
) -> pd.DataFrame:
    """
    Calculate average return, std, annualized return, annualized std, sharpe ratio for the last day of a period within each regime.
    
    Args:
        df (DataFrame): the data only with columns (`'DATE'`, `'regime'`, ...) (the other columns are returns).
        freq_alias (str | None): alias for the length of the return period for average return and std. This can be `'W'`, `'2W'`, `'ME'`. 
            Default is `None` for 1 day. Full list see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
        freq_length (int): length of the return period (in days). Default to `1`.

    Returns:
        results (dict): dict with regimes as keys and dataframe as values. The dataframe has indices as indexes and metrics as columns
    """
    
    # get the last day of each period
    df = df[df['regime'] != df['regime'].shift(-1)]

    results = df.groupby('regime').apply(
        calculate_return_metrics,
        freq_alias = freq_alias,
        freq_length = freq_length,
        include_groups=False
    ).reset_index(names=['regime', 'index'])

    return results


def calculate_returns_forward(
        df: pd.DataFrame, 
        period: int
):
    """
    Calculate returns in the following `period` days for each regime.
    
    Args:
        df (DataFrame): the data with columns (`'DATE'`, `'regime'`, ...) (the other columns are returns).
        period (int): length of the period.

    Returns:
        results: dataframe with indices as indexes, regime as columns and returns as rows.
    """

    returns_cum = df.sort_values(by='DATE', ascending=False).drop(columns=['DATE', 'regime']).rolling(period).sum().shift(1)
    returns_cum = pd.merge(df[['DATE','regime']], returns_cum, left_index=True, right_index=True).dropna().sort_values(by='DATE', ascending=True)

    returns_forward_regimes_grouped = returns_cum.drop(columns='DATE').groupby('regime')

    returns_forward_regimes_stats = pd.DataFrame({
        'mean': returns_forward_regimes_grouped.mean().stack(),
        'std' : returns_forward_regimes_grouped.std().stack()
    }).reset_index(names=['regime', 'index'])

    return returns_forward_regimes_stats


def smooth_regimes(regimes: np.ndarray, window: int | None = 10):
    """
    Smooth the regimes, by assigning the most frequent regime within the sliding window.
    
    Args:
        regimes (ndarray): the regime identification results.
        window (int): length of the sliding window.

    Returns:
        regimes_series (Series): Series of the smoothed regimes.
    """

    regimes_series = pd.Series(regimes)
    regimes_series = regimes_series.rolling(window, min_periods=1).apply(lambda x : x.mode()[0])
    regimes_series = regimes_series.round()

    return regimes_series


def calculate_transition_matrix(regimes_df: pd.DataFrame, at_regimes_level: bool | None = True) -> pd.DataFrame:
    """
    Calculate transition matrix.
    
    Args:
        regimes_df (DataFrame): DateFrame of the regimes which must include columns(`'DATE'`, `'regime'`)
        at_regimes_level (bool): If to calculate regime at regime level. If `'True'`, the probability between the same regime would be zero.
            If `'False'`, the transition matrix is at the data level.
    """
    # ensure the regimes are in ascending order of the dates
    regimes_df = regimes_df.sort_values(by='DATE', ascending=True)
    regimes_df = regimes_df[['DATE', 'regime']]

    if at_regimes_level:
        # keep the rows whose 'regime' is a new one, compared with the previous one
        regimes_df = regimes_df[regimes_df['regime'] != regimes_df['regime'].shift(1)]

    # calculate trainsition matix
    regimes_df['regime_next'] = regimes_df['regime'].shift(-1)
    regimes_df = regimes_df.dropna() # droup the last row
    transition_counts = pd.crosstab(regimes_df['regime'], regimes_df['regime_next'])
    transition_matrix = transition_counts.div(transition_counts.sum(axis=1),axis=0)

    return transition_matrix


def calculate_rwo_entropy(row: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate entropy for the row of probabilities.
    
    Args:
        row (ndarray | Series): row of probabilities.

    Returns:
        entropy (float): entropy of the row.
    """
    row = row[row>0]
    return -np.sum(row * np.log(row))


def calculate_transition_matrix_entropy(matrix: pd.DataFrame, apply_norm: bool | None = False) -> float:
    """
    Calculate the entropy for this trainsition matrix.
    
    Args:
        matrix (DataFrame): the transition matrix, with regimes as indexes and regimes as columns.
            Value at row i column j means the probability of regime i switching to regime j.
        apply_norm: If `Ture`, normalize the entropy by number_of_rows x log(number_of_rows).

    Returns:
        entropy (float): entropy of the matrix.
    """

    entropies_rows = np.apply_along_axis(calculate_rwo_entropy, 1, matrix)
    if apply_norm:
        return np.sum(entropies_rows) / np.log(len(entropies_rows)-1) / len(entropies_rows)
    else:
        return np.sum(entropies_rows)
    

def analyse_regimes_durations(regimes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mean, std, 25, 50, 70 quantiles for regime durations and the total days and 
        the corresponding percentage for each regime and all regime periods.
    
    Args: 
        regimes_df (DataFrame): DataFrame with columns (`'DATE'`, `'regime'`). Other columns will be ignored.

    Returns:
        regimes_durations_stats (DataFrame): DataFrame with columns 
            (`'regime'`, `'mean'`, `'std'`, `'count'`, `'25 quantile'`, `'median'`, `'75 quantile'`, `'percent'`).
    """
    regimes_df = regimes_df[['DATE','regime']]
    regimes_durations = calculate_regime_durations(regimes_df)
    regimes_durations_stats = calculate_regime_durations_statistics(regimes_durations)

    return regimes_durations_stats
