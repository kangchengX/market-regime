import os
import numpy as np
import pandas as pd
from typing import List, Union
from constant import SECURITY, COLUMN_MAP_RETURN
from utils import load_df


def determin_dates(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Determin the dated needed for the final data.
    Rule: 
        1, find (start date, end date) during which all dfs have data.
        2, during this period, find the dates when at least one df has data.

    Args:
        dfs (list): List of the dataframe which has column `'DATE'`.

    Returns:
        common_date (DataFrame): The dataframe with only one column `'DATE'` which is the common dates for all indices.
    """

    if len(dfs) == 1:
        return dfs

    # start date and end date
    date_min = max([df['DATE'].iloc[-2] for df in dfs])
    date_max = min([df['DATE'].iloc[0] for df in dfs])
    
    #  find the dates when at least one df has data
    common_date=pd.merge(dfs[0]['DATE'],dfs[1]['DATE'],how='outer')
    for df in dfs[2:]:
        common_date = pd.merge(common_date,df['DATE'],how='outer')

    # combine the two rules
    common_date = common_date[(date_min<=common_date['DATE']) & (common_date['DATE']<=date_max)]

    return common_date


def generate_data(df: pd.DataFrame, dates: pd.DataFrame):
    """
    Deal with the missing data, i.e. NA.
    Rules:
        1, For the missing prices in the dates, fill them with the last day's price
        2, fill other missing data with 0
    
    Args:
        df (DataFrame): The DataFrame containing columns (`'PX_LAST'`, `'PX_OPEN'`, '`PX_HIGH'`, `'PX_LOW'`).
            It can contain other columns.
        dates (DataFrame): The common dates only with column (`'DATE'`).

    Returns:
        df (DataFrame): The data with missing data filled, with indice name as the Name.
    """

    name = df.Name
    df = pd.merge(df,dates ,how='outer').sort_values(by='DATE',ascending=False)
    df[['PX_LAST', 'PX_OPEN', 'PX_HIGH', 'PX_LOW']] = df[['PX_LAST', 'PX_OPEN', 'PX_HIGH', 'PX_LOW']].bfill()
    df.fillna(0,inplace=True)
    df.Name = name

    return df


def generate_columns(df: pd.DataFrame, dates: pd.DataFrame):
    """
    Calculate columns needed - perctange change, log return, difference.
    For security, calculate PRICE_CUM.
    
    Args:
        df (DataFrame): The loaded dataframe, at least with columns ('PX_LAST') and ('PX_LAST', 'DAY_TO_DAY_TOT_RETURN_GROSS_DVDS') for securites.
        dates (DataFrame): The common dates only with column (`'DATE'`).
    """
    # (v_t / v_{t-1} - 1) * 100, v is PX_LAST
    df['PERCENT'] = (df['PX_LAST'] / df['PX_LAST'].shift(-1) - 1) * 100
    # v_t - v_{t-1}, v is PX_LAST, v is PX_LAST
    df['DIFF'] = df['PX_LAST'] - df['PX_LAST'].shift(-1)

    if df.Name in SECURITY:
        # log(1 + v_t / 100) v is DAY_TO_DAY_TOT_RETURN_GROSS_DVDS
        df['LOG'] = np.log(1 + df['DAY_TO_DAY_TOT_RETURN_GROSS_DVDS'] / 100)
        log_cum = df['LOG'].iloc[::-1].cumsum().iloc[::-1]
        df['PRICE_CUM'] = df['PX_LAST'].iloc[-1] * np.exp(log_cum)
    else:
        # log(v_t / v_{t-1})
        df['LOG'] = np.log(df['PX_LAST'] / df['PX_LAST'].shift(-1))

    name = df.Name
    df = pd.merge(df,dates ,how='right').sort_values(by='DATE',ascending=False)
    df.Name = name

    return df


def clean_data(folder_raw_data: str, folder_new_data: str | None = 'data'):
    """
    Clean the data.
    
    Args:
        folder_raw_data (str): The folder containing the raw data in .csv files.
        folder_new_data (str): The folder to save the cleaned data. Default to `'data'`.

    Files Created:
        cleaned data in .csv files. Dates for the file are the common dates. Filename is the name of the index, and has column `'DATE'`.
    """

    filenames = os.listdir(folder_raw_data)
    filenames = [os.path.join(folder_raw_data, filename) for filename in filenames]
    dfs = [load_df(filename, from_raw=True) for filename in filenames]
    dates = determin_dates(dfs)

    for df in dfs:
        df = generate_data(df, dates)
        df = generate_columns(df, dates)
        df.to_csv(os.path.join(folder_new_data,df.Name) + '.csv', index=False)


def gather_data(
    dfs: List[pd.DataFrame], 
    columns_map: str | dict | None = COLUMN_MAP_RETURN,
    keep_prefix: bool | None = True,
    filename : str | None = None,
    sort_ascending: bool | None = True
) -> pd.DataFrame:
    """
    Generate the final table to use as the inputs of the regime identification model.
    
    Args:
        dfs (list): List of DataFrame of indices data
        columns_map (str|dict): the mapping of the columns to use. Can be:
            str: use columns with the name `columns_map`.
            dict: {index, column_name}, for each index, use the column mapped by the dict.
            Default to `COLUMN_MAP_RETURN`.
        keep_prefix (bool): If `True`, keep the prefix of the column. Default to `True`.
        filename (str): filename to save the model. If None, the filenme will not be saved. Default to `None`.
        sort_ascending (bool): If `True`, sort the values according to dates in ascending order. Default to `True`.
    """

    df_final = dfs[0][['DATE']].copy()

    if type(columns_map) is dict:
        columns_map_func = lambda x : columns_map[x]
    else:
        columns_map_func = lambda x : columns_map

    for df in dfs:
        column_name_sub = columns_map_func(df.Name)
        if keep_prefix:
            column_name_final = column_name_sub + ' ' + df.Name
        else:
            column_name_final = df.Name
        df_final[column_name_final] = df[column_name_sub]

    if sort_ascending:
        df_final = df_final.sort_values('DATE')

    if filename is not None:
        df_final.to_csv(filename,index=False)

    return df_final
