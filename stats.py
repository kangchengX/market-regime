import numpy as np
import pandas as pd
from typing import Dict, Union

# def calculate_regime_durations(regimes: np.ndarray) -> Dict[int, np.ndarray]:
#     '''Calculate regime durations.

#     Args:
#         regimes: regime labels for each time stamp

#     Returns:
#         regimes_durations: dict with reimges as keys and arrays of durations as values
#     '''

#     change_indices = np.where(np.diff(regimes) != 0)[0] + 1
#     durations = np.diff(np.concatenate(([0],change_indices,[len(regimes)])))
#     # the coresponding regime for that duration
#     durations_regimes = np.concatenate((regimes[0:1],regimes[change_indices]))

#     regimes_durations = {regime:durations[durations_regimes==regime] for regime in np.unique(regimes)}

#     return regimes_durations


def calculate_regime_durations(regimes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate regime durations.
    
    Args:
        regimes_df (DataFrame): DateFrame that includes columns ('DATE', 'regime').

    Return:
        regime_durations (DataFrame): DateFrame only with columns ('regime', 'period', 'duration').
    """
    df_inter = regimes_df[['DATE', 'regime']]
    df_inter['regime_if_new'] = df_inter['regime'] != df_inter['regime'].shift(1)
    df_inter['period'] = df_inter['regime_if_new'].cumsum()

    regime_durations = df_inter.groupby(by=['regime', 'period']).size().reset_index(name='duration')

    return regime_durations


def calculate_regime_durations_statistics(regime_durations: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mean, std and quantiles for regime durations for each regime and the whole period
    
    Args:
        regimes_durations (DataFrame): dataframe that includes columns ('regime', 'duration'). Note, the duration here is the duration of the period when the regime keeps the same. Not the total duration.

    Returns:
        statistics (DataFrame): dataframe with columns ('regime', 'mean','std','count','25 quantile', 'median', '75 quantile', 'percent')
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
        freq_length: int | None = 1):
    '''Calculate average return, std, annualized return, annualized std, sharpe ratio
    
    Args:
        df: the data with and only with columns (DATE, .......) (the other columns are returns)
        freq_alias: alias for the length of the period, can be 'W', '2W', 'ME', default is None for 1 day
            full list see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        freq_length: length of the period (in days)

    Return:
        metrics: dataframe with indices as indices and metrics as columns
    '''

    # group by frequency
    if freq_alias is not None:
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
    '''Calculate average return, std, annualized return, annualized std, sharpe ratio within each regime
    
    Args:
        df: the data with and only with columns (DATE, 'regime', ...) (the other columns are returns)
        regimes_df: DateFrame of the regimes with columns('DATE', 'regime')
        freq_alias: alias for the length of the period, can be 'W', '2W', 'ME'. default is None for 1 day
            full list see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        freq_length: length of the period (in days)

    returns:
        results: dict with regimes as keys as dataframe as values. The dataframe has indices as indices and metrics as columns

    '''

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
    '''Calculate average return, std, annualized return, annualized std, sharpe ratio for the last day of a period within each regime
    
    Args:
        df: the data with and only with columns ('DATE', 'regime', ...) (the other columns are returns)
        regimes_df: DateFrame of the regimes with columns('DATE', 'regime')
        freq_alias: alias for the length of the period, can be 'W', '2W', 'ME'. default is None for 1 day
            full list see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        freq_length: length of the period (in days)

    returns:
        results: dict with regimes as keys and dataframe as values. The dataframe has indices as indices and metrics as columns

    '''
    
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
    '''Calculate returns in the following period days after each regime
    
    Args:
        df: the data with columns (DATE, .......) (the other columns are returns)
        regimes: the regime labels for each time stamp
        period: length of the period

    Returns:
        results: dataframe with indices as indices, regime as columns and returns as rows
    '''

    returns_cum = df.sort_values(by='DATE', ascending=False).drop(columns=['DATE', 'regime']).rolling(period).sum().shift(1)
    returns_cum = pd.merge(df[['DATE','regime']], returns_cum, left_index=True, right_index=True).dropna().sort_values(by='DATE', ascending=True)

    returns_forward_regimes_grouped = returns_cum.drop(columns='DATE').groupby('regime')

    returns_forward_regimes_stats = pd.DataFrame({
        'mean': returns_forward_regimes_grouped.mean().stack(),
        'std' : returns_forward_regimes_grouped.std().stack()
    }).reset_index(names=['regime', 'index'])

    return returns_forward_regimes_stats


def smooth_regimes(regimes: np.ndarray, window: int | None = 10):
    '''Smooth the regimes, by assign the most frequent regime within the sliding window
    
    Args:
        regimes: the regime identification results
        window: length of the sliding window

    Returns:
        
        
    '''
    regimes_series = pd.Series(regimes)
    regimes_series = regimes_series.rolling(window,min_periods=1).apply(lambda x : x.mode()[0])
    regimes_series = regimes_series.round()

    return regimes_series


def calculate_transition_matrix_regime_level(regimes_df: pd.DataFrame) -> pd.DataFrame:
    '''Calculate transition matrix at regime level
    
    Args:
        regimes_df: DateFrame of the regimes which must include columns('DATE', 'regime')
    '''

    return calculate_transition_matrix(regimes_df, at_regimes_level=True)


def calcualte_trainsition_matrix_date_level(regimes_df: pd.DataFrame) -> pd.DataFrame:
    '''Calculate transition matrix at date level
    
    Args:
        regimes_df: DateFrame of the regimes which must include columns('DATE', 'regime')
    '''

    return calculate_transition_matrix(regimes_df, at_regimes_level=False)


def calculate_transition_matrix(regimes_df: pd.DataFrame, at_regimes_level: bool | None = True) -> pd.DataFrame:
    '''Calculate transition matrix
    
    Args:
        regimes_df: DateFrame of the regimes which must include columns('DATE', 'regime')
        at_regimes_level: if calculate regime at regime level. If true, the probability between the same regime would be zero
    '''
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
    '''Calculate entropy for the rwo
    
    Args:
        row: row of probabilities

    Returns:
        entropy
    '''
    row = row[row>0]
    return -np.sum(row * np.log(row))


def calculate_transition_matrix_entropy(matrix: pd.DataFrame, apply_norm: bool | None = False) -> float:
    '''Calculate the entropy for this trainsition matrix:
    
    Args:
        matrix: the transition matrix
        apply_norm: If Ture, average the entropies of rows

    Returns:
        entropy
    '''
    entropies_rows = np.apply_along_axis(calculate_rwo_entropy, 1, matrix)
    if apply_norm:
        return np.sum(entropies_rows) / np.log(len(entropies_rows)) / len(entropies_rows)
    else:
        return np.sum(entropies_rows)
    

def analyse_regimes_durations(regimes_df: pd.DataFrame) -> pd.DataFrame:
    '''Analyse statistics of regime durations
    
    Args: 
        regimes_df: dataframe must include columns('DATE', 'regime')

    Returns:
        regimes_durations_stats: dataframe with columns ('regime', 'mean','std','count','25 quantile', 'median', '75 quantile', 'percent')
    '''
    regimes_df = regimes_df[['DATE','regime']]
    regimes_durations = calculate_regime_durations(regimes_df)
    regimes_durations_stats = calculate_regime_durations_statistics(regimes_durations)

    return regimes_durations_stats


if __name__ == '__main__':

    regimes = {
        'DATE': pd.date_range(start='2023-01-01', periods=20, freq='D'),
        'regime': np.random.choice([0,1,2,3], 20, replace=True),
        'price1': np.random.random(20),
        'price2': np.random.random(20)
    }

    regimes_df = pd.DataFrame(regimes)

    print(regimes_df)
    calculate_returns_forward(regimes_df, 10)


# print(regimes_df)
# print(calculate_transition_matrix_regime_level(regimes_df))
# print(calcualte_trainsition_matrix_date_level(regimes_df))
# print(calculate_regime_durations(regimes_df))
# print(calculate_regime_durations_statistics((calculate_regime_durations(regimes_df))))