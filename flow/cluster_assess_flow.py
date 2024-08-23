import argparse, os, json
import cluster, utils, process, data
import numpy as np
import pandas as pd
from warnings import warn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # cluster related
    parser.add_argument('--num_cluster', type=int, default=10, help='Number of clusters. Default to `10`.')
    parser.add_argument('--find_optimal', action='store_true', help='Indicate if to find optimal clusters using elbow method. Store `True`. Default to `False`.')
    parser.add_argument('--pca_varience_keep', type=float, default=0.9, help='Percentage of the varience to keep when selecting components. Default to `0.9`.')
    parser.add_argument('--cluster_type', type=str, default="k-means++", help="Cluster type. Can be `'k-means++'` or `'hierarchical'` (only for features extracted by correlation methods). Default to `'k-means++'`.")
    
    # feature related
    parser.add_argument('--apply_pca', action='store_true', help='Indicate if to apply pca on the features. Store `True`. Default to `False`.')

    # files saving related
    parser.add_argument('--model_dir', type=str, help='Directory where the model and other related data are saved.')
    parser.add_argument('--returns_file', type=str, default='data/data/returns.csv', help="Path of the .csv file containing the daily returns with indices as columns. Default to `'data/data/returns.csv'`.")
    parser.add_argument('--returns_files_folder', type=str, default='data/data/returns', help="Folder containing the .csv files each of which contains trailing and forward returns for the index. Default to `'data/data/returns'`.")
    parser.add_argument('--save_folder', type=str, help='Folder to save the data.')
    
    args = parser.parse_args()

    # deal with the folder to save the results
    if os.path.exists(args.save_folder):
        warn(f'folder {args.save_folder} already exists', RuntimeWarning)
    else:
        os.makedirs(args.save_folder)

    # load data
    returns_df = utils.load_df(args.returns_file, sort_ascending=True)
    returns_dfs = utils.load_returns_dfs(args.returns_files_folder, sort_ascending=True)

    dates = pd.read_csv(os.path.join(args.model_dir, 'dates.csv'), index_col=0)
    dates['DATE'] = pd.to_datetime(dates['DATE'])

    features = np.load(os.path.join(args.model_dir, 'features.npy'))

    # load config and model args
    with open(os.path.join(args.model_dir, 'config.json'), 'r') as f:
        model_config = json.load(f)

    if os.path.exists(os.path.join(args.model_dir, 'model_args.json')):
        with open(os.path.join(args.model_dir, 'model_args.json'), 'r') as f:
            model_args = json.load(f)
    else:
        model_args = None

    # pca related
    dims_pca = []
    if args.apply_pca:
        dims_pca.append(features.shape[1])
        features = data.apply_pca(features)
        dims_pca.append(features.shape[1])
    
    # cluster
    if args.cluster_type == 'k-means++':
        regimes_df, num_cluster  = cluster.cluster(
            features=features,
            num_cluster=args.num_cluster,
            find_optimal=args.find_optimal,
            dates=dates
        )
    elif args.cluster_type == 'hierarchical':
        regimes_df, num_cluster  = cluster.cluster_similarity(
            similarity=features,
            num_cluster=args.num_cluster,
            find_optimal=args.find_optimal,
            dates=dates
        )
    else:
        raise ValueError(f'Unsupported --cluster_type: {args.cluster_type}. This should be k-means++ or hierarchical')

    # analyse clustering results
    analyser = process.Analyser(returns_df=returns_df, returns_dfs=returns_dfs)
    regimes_df = analyser.analyse(features=features, original_regimes_df=regimes_df, folder=args.save_folder)

    # save results
    if args.find_optimal:
        args.optimal_num_cluster = num_cluster

    config = {
        'model_config' : model_config,
        'model_args' : model_args,
        'cluster_config' : vars(args),
        'pca_info' : dims_pca
    }

    with open(os.path.join(args.save_folder, 'config.json'), 'w') as f:
        json.dump(config,f,indent=6, cls=utils.NumpyEncoder)

    regimes_df.to_csv(os.path.join(args.save_folder, 'regimes.csv'))
