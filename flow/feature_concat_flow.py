import argparse, os, json
import numpy as np
import pandas as pd
from warnings import warn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hyperperameters
    parser.add_argument('--weight', type=float, default=1.0, help='Weight for the features obtained by the correlation methods.')

    # files savining related
    parser.add_argument('--deep_learing_model_folder', type=str, help='Folder containing the dates, config and feature for the deep learning model.')
    parser.add_argument('--correlation_model_folder', type=str, help='Folder containing the dates, config, features for the correlation based method.')
    parser.add_argument('--save_folder', type=str, help='Folder to save the results.')

    args = parser.parse_args()

    # default to deep_learing_model_folder + correlation_model_folder + weight
    if args.save_folder is None:
        _, deep_root = os.path.split(args.deep_learing_model_folder)
        _, correlation_root = os.path.split(args.correlation_model_folder)
        args.save_folder = os.path.join('data/inter data/concat', f'{deep_root} + {correlation_root} + {args.weight}')

    print(args.save_folder)

    # deal with the folder to save the results
    if os.path.exists(args.save_folder):
        warn(f'folder {args.save_folder} already exists')
    else:
        os.makedirs(args.save_folder)

    # load dates and check if they match
    dates_deep = pd.read_csv(os.path.join(args.deep_learing_model_folder, 'dates.csv'), index_col=0)
    dates_correlation = pd.read_csv(os.path.join(args.correlation_model_folder, 'dates.csv'), index_col=0)

    # load config
    with open(os.path.join(args.deep_learing_model_folder, 'config.json'), 'r') as f:
        config_deep = json.load(f)
    with open(os.path.join(args.correlation_model_folder, 'config.json'), 'r') as f:
        config_correlation = json.load(f)

    config = {
        'config_concat': vars(args),
        'config_deep' : config_deep,
        'config_correlation' : config_correlation
    }
    
    # load features
    fearures_deep = np.load(os.path.join(args.deep_learing_model_folder, 'features.npy'))
    fearures_correlation = np.load(os.path.join(args.correlation_model_folder, 'features.npy'))

    dates_deep['index_deep'] = dates_deep.index
    dates_correlation['index_correlation'] = dates_correlation.index
    dates = pd.merge(dates_deep, dates_correlation, on='DATE')

    fearures = np.concatenate((fearures_deep[dates['index_deep'].values], fearures_correlation[dates['index_correlation'].values] * args.weight), axis=-1)

    # save features
    np.save(os.path.join(args.save_folder, 'features.npy'), fearures)

    # save config
    with open(os.path.join(args.save_folder, 'config.json'), 'w') as f:
        json.dump(config, f, indent=6)

    # save dates
    dates[['DATE']].to_csv(os.path.join(args.save_folder, 'dates.csv'))


