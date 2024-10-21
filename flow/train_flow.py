import argparse, time, os, json
import data, process, visualization
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from networks import AutoEncoder
from utils import str_list, load_df
from warnings import warn

if __name__ == '__main__':
    
    current_time_string = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')

    parser = argparse.ArgumentParser()

    # data and data loader related
    parser.add_argument('--window_size', type=int, default=128, help='Length of the sliding window. Default to 128.')
    parser.add_argument('--slide_step', type=int, default=10, help='Length of the sliding step. Default to 10.')
    parser.add_argument('--images_encoding_method', type=str, default='gasf', help='Method to transform univariate time series into an image. Default to gasf.')
    parser.add_argument('--lag', type=int, default=1, help='Lag for the taget image. The image at timestamp t + lag*slide_step will be the target for input image at timestamp t. Default to 1.')
    parser.add_argument('--indices', type=str_list, default='', help='Number of indices. Default to an empty string.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--apply_norm', action='store_true', help='If to normalize the images.')

    # model structure related
    parser.add_argument('--feature_dimension', type=int, default=20, help='Feature dimension. Default to 20.')
    
    # training related
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for the AutoEncoder. Default to 1e-5.')
    parser.add_argument('--num_epochs', type=int, default=3000, help='Number of epochs for the AutoEncoder.')
    parser.add_argument('--optimizer_type', type=str, default='adam', help='Type of the optimizer of the AutoEncoder. Default to adam.')

    # files saving related
    parser.add_argument('--model_save_period', type=int, default=250, help='The frequency to save the models. If k, the model will be saved every k epochs. Default to 250.')
    parser.add_argument('--save_losses_plot',  action='store_true', help='If to save the plot for losses against epochs.')
    parser.add_argument('--save_folder', type=str, default=os.path.join('data/inter data/deep learning', current_time_string), help='Path to save the model. Default to data/inter data/deep learning/current_time_string.')
    parser.add_argument('--data_file_path', type=str, default=r'data\data\prices.csv', help='Path of the price data file. Default to data/data/prices.csv')
    parser.add_argument('--save_encoded_features', action='store_true', help='If to save the extracted features.')


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

    num_indices = df.shape[1] - 1

    timeseries_images, dates = data.timeseries_to_image(
        timeseries=df, 
        window=args.window_size, 
        slide_step=args.slide_step, 
        apply_norm=args.apply_norm,
        method=args.images_encoding_method
    )
    dataset = data.TimeSeriesImagesDataset(timeseries_images, dates=dates, lag=args.lag)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size)

    # build model
    image_width_encoded = int(args.window_size / 2**5)
    cnn_args = {
        'channels_all' : [
            [1,64,64],
            [64,128,128],
            [128,256,256,256],
            [256,512,512,512],
            [512,512,512,512]
        ],
        'mlp_dims' : [512*num_indices*image_width_encoded*image_width_encoded, 1000, 100, args.feature_dimension]
    }

    mlp_args = {
        'dims' : [args.feature_dimension, 1000, 1*(num_indices*args.window_size)*args.window_size]
    }

    out_feature_shape = [1, num_indices*args.window_size, args.window_size]

    model_args = {
        'encoder' : 'cnn',
        'encoder_args' : cnn_args,
        'decoder' : 'mlp',
        'decoder_args' : mlp_args,
        'reshape_encoder' : None,
        'reshape_decoder' : 'unflatten',
        'inter_feature_shape' : None,
        'out_feature_shape' : out_feature_shape
    }

    model = AutoEncoder(**model_args)

    # load processor
    processor = process.Processor(
        model=model,
        criterion=nn.MSELoss(),
        data_loader=data_loader,
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer_type,
        save_folder = args.save_folder
    )
    
    # save args
    with open(os.path.join(args.save_folder, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # save model args
    with open(os.path.join(args.save_folder, 'model_args.json'), 'w') as f:
        json.dump(model_args, f, indent=6)

    # save model overview
    with open(os.path.join(args.save_folder, 'model_overview.txt'), 'w') as f:
        f.write(str(model))
    
    # train the model
    print('-'*10 + 'start training' + '-'*10 )

    processor.train(
        epochs=args.num_epochs,
        save_model=True,
        save_period=args.model_save_period,
        save_log=True
    )

    # save features:
    if args.save_encoded_features:
        processor.extract_feature_representation()

    # save logs
    if args.save_losses_plot:
        visualization.visualize_losses(
            losses=processor.losses,
            filename=os.path.join(args.save_folder, 'losses.png'),
            show_fig=False,
            backend='Agg'
        )

    print('-'*10 + 'finish training' + '-'*10 )
