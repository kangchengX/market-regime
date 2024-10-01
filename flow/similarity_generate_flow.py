import argparse, os, time, json
import data, visualization
from warnings import warn
from datetime import datetime
from utils import str_list, load_df
from process import SimGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data related
    parser.add_argument('--window_size', type=int, default=128)
    parser.add_argument('--slide_step', type=int, default=10)
    parser.add_argument('--correlation_methods', type=str_list, default='pearson,kendall,spearman')
    parser.add_argument('--indices', type=str_list, default='')

    # similarity matrix related
    parser.add_argument('--sim_method', type=str)

    # files saving related
    parser.add_argument('--save_folder', type=str, default=os.path.join(r'data\inter data\correlation', datetime.fromtimestamp(time.time()).strftime(r'%Y-%m-%d %H-%M-%S')))
    parser.add_argument('--data_file_path', type=str, default=r'data\data\returns.csv')

    args = parser.parse_args()

    # deal with the folder to save the results

    if os.path.exists(args.save_folder):
        warn(f'folder {args.save_folder} already exists')
    else:
        os.makedirs(args.save_folder)

    # prepare data
    df = load_df(args.data_file_path, sort_ascending=True)

    if args.indices:
        df = df[['DATE'] + args.indices]
    else:
        args.indices = df.drop(columns='DATE').columns.to_list()

    correlation_matrices, dates = data.timeseries_to_correlation(
        timeseries=df,
        window=args.window_size, 
        slide_step=args.slide_step,
        methods=args.correlaton_methods
    )
    
    dataset = data.TimeSeriesCorrelationsDataset(data=correlation_matrices, dates=dates)

    # generate similarity matrix
    print('start : generate similarity matrix')
    sim_generator = SimGenerator(dataset=dataset, method=args.sim_method)
    sim = sim_generator.generate_features(save_folder=args.save_folder)
    print('finish : generate correlation matrix')

    # save args
    with open(os.path.join(args.save_folder, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # save fig
    print('start : generate image')
    visualization.visualize_similarity_matrix(
        dates=dates, 
        sim=sim, 
        title=args.sim_method, 
        interval=1000 // args.slide_step,
        show=False,
        backend='Agg',
        figsize=(8, 7),
        filename=os.path.join(args.save_folder, 'sim.png')
    )
    print('finish : generate image')
