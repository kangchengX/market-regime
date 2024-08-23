import pandas as pd
import numpy as np
import os, json, inspect, math
from typing import Literal, List
from functools import wraps

def load_df(
        filename: str, 
        from_raw: bool | None = False, 
        date_format: str | None = r"%Y-%m-%d",
        sort_ascending: bool | None = False
) -> pd.DataFrame:
    """
    Load file to dataframe for this project. Covert the column DATE into datatime format.
    
    Args:
        filename: path of the file.
        from_raw: `Ture` means the files are raw files. And the value at (0,0) of the .csv file is the name of the index.\
            False means the file are cleaned or the final gathered table. The dates for each cleaned file as well as \
            the final gathered table are the same and the filename is the name of the df (for index table, it is also the name of the index).
        date_format: format of the date. Default is `r"%Y-%m-%d"`. Will be changed to `r"%d/%m/%Y"` automatically when `from_raw` = `True`.
        sort_ascending: if to sort the values according to the column DATE. Note, if `False`, no sorting operation will be performed, and the order is kept,\
            either ascending or desending. Default to `False.

    Returns:
        index_df: the loaded dataframe, having a column 'DATE' with type of datetime
    """

    index_df = pd.read_csv(filename, header=0)
    if from_raw:
        index_df.Name = index_df.columns[0]
        index_df = index_df.rename(columns={index_df.columns[0]:'DATE'})
        index_df = index_df.dropna(subset='DATE')
        date_format = r"%d/%m/%Y"
    else:
        index_df.Name = os.path.splitext(os.path.basename(filename))[0]

    index_df['DATE'] = pd.to_datetime(index_df['DATE'], format=date_format)

    if sort_ascending:
        index_df = index_df.sort_values('DATE')

    return index_df

def load_returns_dfs(
        folder: str | None = r'data\data\returns',
        sort_ascending: bool | None = True,
) -> List[pd.DataFrame]:
    """
    Load returns files in the folder `folder`.

    Args:
        folder (str): folder containg the returns files. The name of the file with prefix `'performace_'` and extenson `'.csv'`. Each file has columns with names \
            (`'DATE'`, `'level'`, `'d'`, `'d_126_tr'`, `'d_63_tr'`, `'d_21_tr'`, `'d_1_tr'`, `'d_1_fw1'`, `'d_21_fw1'`, `'d_63_fw1'`, `'d_126_fw1'`).
        sort_ascending (bool): `True` indicates sort the rows according.
    """
    dfs = []
    for filename in os.listdir(folder):
        df = pd.read_csv(os.path.join(folder, filename))
        root, _ = os.path.splitext(filename)
        df.Name = root
        df['DATE'] = pd.to_datetime(df['DATE'], format=r"%Y-%m-%d")
        if sort_ascending:
            df.sort_values(by='DATE')
        dfs.append(df)

    return dfs


def to_distance(matrix: np.ndarray | pd.DataFrame):
    """
    Convert the correlation matrix in to distance matrix'.
    Method: \sqrt{2 * (1-|element|)}

    Args:
        matrix (ndarray | DataFrame): the input matrix.

    Returns:
        out (ndarray | DataFrame): the distance matrix.
    """
    return np.sqrt(2 * (1 - np.abs(matrix)))


def append_df_to_excel(
        filename: str, 
        df: pd.DataFrame, 
        sheet_name: str
):
    '''Append the dataframe to the excel'''
    with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        startrow = 0
        if sheet_name in writer.book.sheetnames:
            startrow = writer.book[sheet_name].max_row

        df.to_excel(writer,sheet_name=sheet_name, startrow=startrow)


def fill_dates_values(df: pd.DataFrame, regimes_df:pd.DataFrame):
    df_regimes_all = pd.merge(df, regimes_df, on='DATE', how='left').ffill().dropna()

    return df_regimes_all


def str_bool(bool_str: str) -> bool:
    if bool_str == 'True':
        return True
    elif bool_str == 'False':
        return False
    else:
        raise ValueError(f'Unsupported bool string {bool_str}')
    
def str_list(list_str: str) -> list:
    if list_str:
        return list_str.split(',')
    else:
        return []

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return super().default()
    

def check_value_close(value: int | float | bool | str, values:list):
    if isinstance(value, (str, int, bool)):
        return value in values
    else:
        for value_in_list in values:
            if math.isclose(value_in_list, value):
                return True
    return False

def check_args(args_dict, **kwargs): 
    """
    Check if `args_dict[key]` is in `value` for key, value in kwargs. Notice, if value is None, this will not be check, \
        and if the key in `args_dict` will not be check either. If all values are `None`, `args_dict` will not be used. 
    
    """
    for arg, value in kwargs.items():
        if value is not None:
            value = [value] if isinstance(value, (int, float, bool, str)) else value
            if not check_value_close(args_dict[arg], value):
                return False      
    return True


def filter_correlation_dir(
        parent_folder: str | None = 'data/inter data/correlation',
        return_with_parent_folder: bool | None = True,
        window_size: int | list | None = None,
        slide_step: int | list | None = None,
        num_indices: int | list | None = None,
        num_correlation_methods: int | list | None = None,
        sim_method: str | list | None = None     
):  
    dirs = []
    for model_dir in os.listdir(parent_folder):
        with open(os.path.join(parent_folder, model_dir, 'config.json'), 'r') as f:
            config = json.load(f)

        config['num_correlation_methods'] = len(config["correlation_methods"])
        config['num_indices'] = len(config['indices'])
        if check_args(
            config,
            window_size=window_size,
            slide_step=slide_step,
            num_correlation_methods=num_correlation_methods,
            num_indices=num_indices,
            sim_method=sim_method
        ):
            dirs.append(os.path.join(parent_folder,model_dir) if return_with_parent_folder else model_dir)

    return dirs
    
def filter_deep_model_dir(
        parent_folder: str | None = 'data/inter data/deep learning',
        return_with_parent_folder: bool | None = True,
        window_size: int | list | None = None,
        slide_step: int | list | None = None,
        images_encoding_method: Literal['gasf', 'gadf', 'mtf'] | list | None = None,
        lag: int | list | None = None,
        num_indices: int | list | None = None,
        batch_size: int | list | None = None,
        apply_norm: bool | list | None = None,
        learning_rate: float | list | None = None,
        num_epochs: int | list | None = None,
        optimizer_type: str | list | None = None,
        feature_dimension: int | list | None = None,
        encoder: Literal['cnn','mlp','conv','conv_trans'] | list | None = None,
        decoder: Literal['cnn','mlp','conv','conv_trans'] | list | None = None,
):  
    dirs = []
    for model_dir in os.listdir(parent_folder):

        with open(os.path.join(parent_folder, model_dir, 'config.json'), 'r') as f:
            config = json.load(f)

        with open(os.path.join(parent_folder, model_dir, 'model_args.json'), 'r') as f:
            model_args = json.load(f)

        config['num_indices'] = len(config['indices'])

        if check_args(
            config,
            window_size=window_size,
            slide_step=slide_step,
            images_encoding_method=images_encoding_method,
            lag=lag,
            num_indices=num_indices,
            batch_size=batch_size,
            apply_norm=apply_norm,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            optimizer_type=optimizer_type,
            feature_dimension=feature_dimension
        ) and check_args(
            model_args,
            encoder=encoder,
            decoder=decoder
        ):
            dirs.append(os.path.join(parent_folder,model_dir) if return_with_parent_folder else model_dir)

    return dirs


def filter_end_to_end_dir(
        parent_folder: str | None = 'data/inter data/end-to-end',
        return_with_parent_folder: bool | None = True,
        window_size: int | list | None = None,
        slide_step: int | list | None = None,
        images_encoding_method: Literal['gasf', 'gadf', 'mtf'] | list | None = None,
        lag: int | list | None = None,
        num_indices: int | list | None = None,
        batch_size: int | list | None = None,
        apply_norm: bool | list | None = None,
        learning_rate: float | list | None = None,
        num_epochs: int | list | None = None,
        optimizer_type: str | list | None = None,
        feature_dimension: int | list | None = None,
        l2_reg_weight: float | list | None = None,
        entropy_reg_weight: float | list | None = None
):  
    dirs = []
    for model_dir in os.listdir(parent_folder):

        with open(os.path.join(parent_folder, model_dir, 'config.json'), 'r') as f:
            config = json.load(f)

        config['num_indices'] = len(config['indices'])

        if check_args(
            config,
            window_size=window_size,
            slide_step=slide_step,
            images_encoding_method=images_encoding_method,
            lag=lag,
            num_indices=num_indices,
            batch_size=batch_size,
            apply_norm=apply_norm,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            optimizer_type=optimizer_type,
            feature_dimension=feature_dimension,
            l2_reg_weight=l2_reg_weight,
            entropy_reg_weight=entropy_reg_weight
        ):
            dirs.append(os.path.join(parent_folder,model_dir) if return_with_parent_folder else model_dir)

    return dirs

def filter_concat_dit(
        parent_folder: str | None = 'data/inter data/concat',
        return_with_parent_folder: bool | None = True,
        weight: float | list | None = None
):
    dirs = []
    for model_dir in os.listdir(parent_folder):

        with open(os.path.join(parent_folder, model_dir, 'config.json'), 'r') as f:
            config = json.load(f)

        if check_args(config, weight):
            dirs.append(os.path.join(parent_folder,model_dir) if return_with_parent_folder else model_dir)

    return dirs


def filter_results_dir(
        parent_folder: str,
        depth_is_1: bool,
        return_with_parent_folder: bool | None = True,
        window_size: int | list | None = None,
        slide_step: int | list | None = None,
        images_encoding_method: Literal['gasf', 'gadf', 'mtf'] | list | None = None,
        lag: int | list | None = None,
        num_indices: int | list | None = None,
        batch_size: int | list | None = None,
        apply_norm: bool | list | None = None,
        learning_rate: float | list | None = None,
        num_epochs: int | list | None = None,
        optimizer_type: str | list | None = None,
        feature_dimension: int | list | None = None,
        encoder: Literal['cnn','mlp','conv','conv_trans'] | list | None = None,
        decoder: Literal['cnn','mlp','conv','conv_trans'] | list | None = None,
        num_cluster: int | list | None = None,
        apply_pca: bool | list | None = None,
        find_optimal: bool | list | None = None,
        num_correlation_methods: int | list | None = None,
        sim_method: str | list | None = None,
        cnn_depth: int | list | None = None,
        num_codes: int | list | None = None,
        entropy_reg_weight: float | list | None = None,
        l2_reg_weight: float | list | None = None,
        concat_weight: float | list | None = None
):
    dirs = []
    for dir in os.listdir(parent_folder):
        if not os.path.isdir(os.path.join(parent_folder, dir)):
            continue
        if depth_is_1:
            dir = ''
        for child_dir in os.listdir(os.path.join(parent_folder, dir)):
            child_dir_full_relative_path = os.path.join(parent_folder, dir, child_dir)
            if not os.path.isdir(child_dir_full_relative_path):
                continue
            try:
                with open(os.path.join(child_dir_full_relative_path, 'config.json'), 'r') as f:
                    config = json.load(f)
            except:
                print('Unable to open {}'.format(os.path.join(child_dir_full_relative_path, 'config.json')))
                continue

            config["model_config"]['num_indices'] = len(config["model_config"].get('indices', []))
            config["model_config"]['num_correlation_methods'] = len(config["model_config"].get("correlaton_methods", []))
            config["model_config"]['concat_weight'] = config["model_config"].get("config_concat", {}).get("weight", 0)
                
            if check_args(
                config["model_config"],
                window_size=window_size,
                slide_step=slide_step,
                images_encoding_method=images_encoding_method,
                lag=lag,
                num_indices=num_indices,
                batch_size=batch_size,
                apply_norm=apply_norm,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                optimizer_type=optimizer_type,
                feature_dimension=feature_dimension,
                num_correlation_methods=num_correlation_methods,
                sim_method=sim_method,
                cnn_depth=cnn_depth,
                num_codes=num_codes,
                l2_reg_weight=l2_reg_weight,
                entropy_reg_weight=entropy_reg_weight

            ) and check_args(
                config["model_args"],
                encoder=encoder,
                decoder=decoder
            ) and check_args(
                config["cluster_config"],
                num_cluster=num_cluster,
                find_optimal=find_optimal,
                apply_pca=apply_pca
            ):
                if return_with_parent_folder:
                    dirs.append(child_dir_full_relative_path)
                else:
                    dirs.append(os.path.join(dir, child_dir))

        if depth_is_1:
            return dirs
    
    return dirs

def gather_metircs(dirs: list):
    dfs = []
    for dir in dirs:
        with open(os.path.join(dir, 'analysis.json'), 'r') as f:
            analysis = json.load(f)
        with open(os.path.join(dir, 'config.json'), 'r') as f:
            config = json.load(f)
        df = pd.DataFrame(analysis['cluster_ana_on_returns'])
        for metric, value in analysis["cluster_ana"].items():
            df[metric] = value
        df['window_size'] = config['model_config']['window_size']
        df['num_indices'] = len(config['model_config']['indices'])
        df['feature_dimension']  = config['model_config']['feature_dimension']
        df['apply_pca'] = config['cluster_config']['apply_pca']
        df['num_cluster'] = config['cluster_config']['num_cluster']
        dfs.append(df)
    
    return pd.concat(dfs)
    
if __name__ == '__main__':
    # dir = r"F:\code\results\2024-07-20 03-14-47\3 - with pca - without elbow method"

    # with open(os.path.join(dir, 'analysis.json'), 'r') as f:
    #     analysis = json.load(f)
    # with open(os.path.join(dir, 'config.json'), 'r') as f:
    #     config = json.load(f)
    # df = pd.DataFrame(analysis['cluster_ana_on_returns'])
    # for metric, value in analysis["cluster_ana"].items():
    #     df[metric] = value
    # df['window_size'] = config['model_config']['window_size']
    # df['num_indices'] = len(config['model_config']['indices'])
    # df['feature_dimension']  = config['model_config']['feature_dimension']
    # df['apply_pca'] = config['cluster_config']['apply_pca']
    # df['num_cluster'] = config['cluster_config']['num_cluster']

    # print(df)

    lists = [1,2,3,4]

    # test = 5
    # print(check_value_close(test, lists))