import argparse, os, json, time
import utils, process, data, networks, loss
from warnings import warn
from datetime import datetime
from torch.utils.data import DataLoader


if __name__ == '__main__':

    current_time_string = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')

    parser = argparse.ArgumentParser()

    # data and data loader related
    parser.add_argument('--window_size', type=int, default=128)
    parser.add_argument('--slide_step', type=int, default=10)
    parser.add_argument('--images_encoding_method', type=str, default='gasf')
    parser.add_argument('--lag', type=int, default=1)
    parser.add_argument('--indices', type=utils.str_list, default='')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--apply_norm', action='store_true')

    # model structure related
    parser.add_argument('--cnn_depth', type=int, default=4)
    parser.add_argument('--feature_dimension', type=int, default=1000)
    parser.add_argument('--num_codes', type=int, default=10)

    # training related
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--l2_reg_weight', type=float, default=0.0)
    parser.add_argument('--entropy_reg_weight', type=float, default=1.0)
    
    # files saving related
    parser.add_argument('--model_folder', type=str, default=os.path.join('data/inter data/end-to-end', current_time_string))
    parser.add_argument('--results_folder', type=str, default=os.path.join('results/end-to-end', current_time_string))
    parser.add_argument('--data_file', type=str, default=r'data\data\prices.csv')
    parser.add_argument('--returns_file', type=str, default=r'data\data\returns.csv')
    parser.add_argument('--returns_files_folder', type=str, default=r'data\data\returns')


    args = parser.parse_args()

    # deal with the folder to save the results
    if os.path.exists(args.model_folder):
        warn(f'folder {args.model_folder} already exists')
    else:
        os.makedirs(args.model_folder)

    if os.path.exists(args.results_folder):
        warn(f'folder {args.results_folder} already exists')
    else:
        os.makedirs(args.results_folder)

    # prepare data
    prices_df = utils.load_df(args.data_file, sort_ascending=True)
    returns_df = utils.load_df(args.returns_file, sort_ascending=True)
    returns_dfs = utils.load_returns_dfs(args.returns_files_folder)

    if args.indices:
        prices_df = prices_df[['DATE'] + args.indices]
    else:
        args.indices = prices_df.drop(columns='DATE').columns.to_list()

    num_indices = prices_df.shape[1] - 1

    timeseries_images, dates = data.timeseries_to_image(
        timeseries=prices_df, 
        window=args.window_size, 
        slide_step=args.slide_step, 
        apply_norm=args.apply_norm,
        method=args.images_encoding_method
    )
    dataset = data.TimeSeriesImagesDataset(timeseries_images, dates=dates, lag=args.lag)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size)
    
    # build model
    image_width_encoded = int(args.window_size / 2**args.cnn_depth)
    cnn_channels = [
        [1,64,64],
        [64,128,128],
        [128,256,256,256],
        [256,512,512,512],
        [512,512,512,512] 
    ][:args.cnn_depth]

    cnn_final_channels = cnn_channels[args.cnn_depth-1][-1]

    cnn_args = {
        'channels_all' : cnn_channels,
        'mlp_dims' : [cnn_final_channels*num_indices*image_width_encoded*image_width_encoded, args.feature_dimension]
    }

    model_args = {
        'encoder' : 'cnn',
        'encoder_args' : cnn_args,
        'num_features' : args.feature_dimension,
        'num_codes' : args.num_codes,
        'out_activation' : 'softmax'
    }

    model = networks.AutoEncoderCodeBook(**model_args)

    # load processor
    processor = process.Processor(
        model=model,
        criterion=loss.KLDivergenceLoss(l2_reg_weight=args.l2_reg_weight, entropy_reg_weight=args.entropy_reg_weight),
        data_loader=data_loader,
        learning_rate=args.learning_rate,
        optimizer_type='adam',
        save_folder=args.model_folder
    )

    # save args
    with open(os.path.join(args.model_folder, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # save model args
    with open(os.path.join(args.model_folder, 'model_args.json'), 'w') as f:
        json.dump(model_args, f, indent=6)

    # save model overview
    with open(os.path.join(args.model_folder, 'model_overview.txt'), 'w') as f:
        f.write(str(model))
    
    # save config in results folder
    results_config = {
        "model_config" : vars(args),
        "model_args" : model_args,
        "cluster_config" : {
            'num_cluster' : args.num_codes
        }
    }
    
    # train the model
    print('-'*10 + 'start training' + '-'*10 )
    processor.train(
        epochs=args.num_epochs,
        save_model=False,
        save_period=None,
        save_log=True,
        siamese=True
    )
    print('-'*10 + 'finish training' + '-'*10 )

    # extract regimes
    regimes_df = processor.extract_regimes()

    # analyse clustering results
    analyser = process.Analyser(returns_df=returns_df, returns_dfs=returns_dfs)
    regimes_df = analyser.analyse(features=None, original_regimes_df=regimes_df, folder=args.results_folder)

    with open(os.path.join(args.results_folder, 'config.json'), 'w') as f:
        json.dump(results_config, f, indent=6, cls=utils.NumpyEncoder)

    regimes_df.to_csv(os.path.join(args.results_folder, 'regimes.csv'))
    