import subprocess, os, time
import utils
from datetime import datetime
from constant import PYTHON_INTER


def execute_similarity_generate_flow(parent_folder: str):
    """
    Execute the first part of experiment one, i.e., feature extraction based on correlation methods.

    Hyperparameter tuning:
        indices, sim method, correlation methods, window_sizes, sliding step, total = 3 x 2 x 2 x 2 x 2 = 48

    Args:
        parent_folder: Folder to save the results in, containing child folders, each of which contains `config.json`, `dates.csv`, `features.npy`, `sim.png`.
    """

    # tune hyperparamters of indices, sim_method, corr_methods, window_size and slide step
    indices_list = [
        ['MXWO Index', 'VIX Index', 'CSI BARC Index'],
        ['MXWO Index', 'VIX Index', 'CSI BARC Index', 'USGG10YR Index', 'DXY Curncy'],
        ''
    ]
    sim_methods = ['meta', 'cophenetic']
    corr_methods_list = [['pearson'], ['pearson', 'kendall', 'spearman']]
    window_sizes = [64, 128]
    slide_steps = [1, 10]

    for indices in indices_list:
        for sim_method in sim_methods:
            for corr_methods in corr_methods_list:
                for window_size in window_sizes:
                    for slide_step in slide_steps:
                        command = [
                            PYTHON_INTER,
                            "flow/similarity_generate_flow.py",
                            '--window_size', f'{window_size}',
                            '--slide_step', f'{slide_step}',
                            '--correlation_methods', ','.join(corr_methods),
                            '--indices', ','.join(indices),
                            '--sim_method', f'{sim_method}',
                            '--save_folder', os.path.join(parent_folder, datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S'))
                        ]

                        subprocess.run(command)


def execute_train_flow(parent_folder: str):
    """
    Execute the first part of experiment two, i.e., feature extraction using cnn autoencoder.

    Hyperparameter tuning:
        learning_rate, window_size, indices, total = 3 x 2 x 2 = 12.
        feature_dimension, total = 5.
        Total = 12 + 5 - 1 = 16
    
    Args:
        parent_folder: folder to save the results in, containing child folders, each of which contains `config.json`, `dates.csv`, `features.npy`, `losses.png`, \
            multiple `model - {epochs}.pth`, `model_args.json`, `model_overview.txt`, and `training_log.json`.
    """

    learning_rates = [1e-4, 1e-5]
    window_sizes = [64, 128]
    indices_list = [
        ['MXWO Index', 'VIX Index', 'CSI BARC Index'],
        ['MXWO Index', 'VIX Index', 'CSI BARC Index', 'USGG10YR Index', 'DXY Curncy'],
        ''
    ]

    # tune hyperparameters of learning_rate, window_size, indices,
    # with slide_step = 10, images_encoding_method = 'gasf', lag = 1, batch_size = 32, feature_dimension = 20, num_epochs = 3000
    for learning_rate in learning_rates:
        for window_size in window_sizes:
            for indices in indices_list:
                subprocess.run([
                    PYTHON_INTER,
                    'flow/train_flow.py',
                    '--window_size', f'{window_size}',
                    '--indices', ','.join(indices),
                    '--learning_rate', f'{learning_rate}',
                    '--save_folder', os.path.join(parent_folder, datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')),
                    '--apply_norm', '--save_losses_plot', '--save_encoded_features'
                ])

    feature_dimensions = [10, 30, 40, 50]
    indices = ['MXWO Index', 'VIX Index', 'CSI BARC Index', 'USGG10YR Index', 'DXY Curncy']
    
    # tune hyperparameters of feature_dimension
    # with window_size = 128, slide step = 10, images_encoding_method = 'gasf', lag = 1, len(indices) = 5, batch_size = 32, learning_rate = 1e-5, num_epochs = 3000
    for feature_dimension in feature_dimensions:
        subprocess.run([
            PYTHON_INTER,
            'flow/train_flow.py',
            '--feature_dimension', f'{feature_dimension}',
            '--indices', ','.join(indices),
            '--save_folder', os.path.join(parent_folder, datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')),
            '--apply_norm', '--save_losses_plot', '--save_encoded_features'  
        ])



def execute_feature_concat_flow(deep_learing_model_folder: str, correlation_model_folder: str, parent_folder: str):
    """
    Execute combiations of experiment one and experiment two.

    Hyperparameter tuning:
        weight, total = 4

    Args:
        deef_learning_model_folder (str): folder where the results of training of autoencoder is, containing child folders, each of which contains `config.json`, `dates.csv`, `features.npy`, `losses.png`, \
            multiple `model - {epochs}.pth`, `model_args.json`, `model_overview.txt`, and `training_log.json`.
        correlation_model_folder (str): folder where the results of similarity matrix extraction results are, containing child folders, each of which contains `config.json`, `dates.csv`, `features.npy`, `sim.png`.
        parent_folder (str): folder to save the results in, containing child folder, each of which contains `config.json`, `dates.csv`, `features.npy`.
    """
    # tune hyperparameters of weight
    weights = [0.1, 1.0, 10, 100]
    _, deep_root = os.path.split(deep_learing_model_folder)
    _, correlation_root = os.path.split(correlation_model_folder)
    for weight in weights:
        subprocess.run([
            PYTHON_INTER,
            'flow/feature_concat_flow.py',
            '--weight', f'{weight}',
            '--deep_learing_model_folder', deep_learing_model_folder,
            '--correlation_model_folder', correlation_model_folder,
            '--save_folder', os.path.join(parent_folder, f'{deep_root} + {correlation_root} + {weight}')
        ])


def execute_cluster_flow(feature_parent_folder: str, add_hierarchical: bool):
    """
    Execute the second part of experiment one or two, i.e., cluster based on features.

    Hyperparameter tuning:
        num_clusters, pca or not, (not find optimal), total = 8 x 2 = 16,
        pca or not (find optimal), total = 2,
        Total = 16 + 2 =18.

    Args:
        parent_folder: folder containing child folders, each of which represents the corresponding model folder and contains child-child folders, \
            each of which represents correspond hyperparamter set and contains `analysis.csv`, `analysis.json`, `config.json`, `durations.png`, \
            `regimes.csv`, `regimes.png`, `transition matrix data level.png`, and `transition matrix regime level.png`.
    """

    _, expirments_name = os.path.split(feature_parent_folder)

    apply_pcas = [False, True]
    num_clusters = [3 ,4, 5, 6, 7, 8, 9, 10]

    # tune hyperparameters of apply_pcas, num_clusters, find_optimal (from 1 to 10)
    # with pca_varience_keep = 0.9, cluster_type =  k-means++
    for dir in os.listdir(feature_parent_folder):
        for alply_pca in apply_pcas:
            # don't find optimal
            for num_cluster in num_clusters:
                suffix = '{} - {} - without elbow method'.format(num_cluster, 'with pca' if alply_pca else 'without pca')
                command = [
                    PYTHON_INTER,
                    "flow/cluster_assess_flow.py",
                    '--num_cluster', f'{num_cluster}',
                    '--model_dir', os.path.join(feature_parent_folder, dir),
                    '--save_folder', os.path.join('results', expirments_name, dir, suffix)
                ]
                if alply_pca:
                    command.append('--apply_pca')
                subprocess.run(command)

            # find optimal from 1 to 10
            suffix = '{} - with elbow method from  1 to 10'.format('with pca' if alply_pca else 'without pca')
            command = [
                PYTHON_INTER,
                "flow/cluster_assess_flow.py",
                '--num_cluster', '10',
                '--model_dir', os.path.join(feature_parent_folder, dir),
                '--save_folder', os.path.join('results', expirments_name, dir, suffix),
                '--find_optimal'
            ]
            if alply_pca:
                command.append('--apply_pca')
            subprocess.run(command)

    # only for similarity matrix
    # tune hyperparameters of num_clusters, find_optimal (from 1 to 10)
    # with pca_varience_keep = 0.9, cluster_type = hierarchical, apply_pca = False, since features should be a matrix
    if add_hierarchical:
        for dir in os.listdir(feature_parent_folder):
            # don't find optimal
            for num_cluster in num_clusters:
                suffix = '{} - h - without elbow method'.format(num_cluster)
                command = [
                    PYTHON_INTER,
                    "flow/cluster_assess_flow.py",
                    '--num_cluster', f'{num_cluster}',
                    '--model_dir', os.path.join(feature_parent_folder, dir),
                    '--save_folder', os.path.join('results', expirments_name, dir, suffix),
                    "--cluster_type", "hierarchical"
                ]
                subprocess.run(command)

            # find optimal from 10
            suffix = 'h - with elbow method from  1 to 10'
            command = [
                PYTHON_INTER,
                "flow/cluster_assess_flow.py",
                '--num_cluster', '10',
                '--model_dir', os.path.join(feature_parent_folder, dir),
                '--save_folder', os.path.join('results', expirments_name, dir, suffix),
                "--cluster_type", "hierarchical",
                '--find_optimal'
            ]
            subprocess.run(command)


def execute_end_to_end_flow(model_parent_folder: str, results_parent_folder: str):
    """
    Execute experiement three.

    Hyperparameter tuning:
        feature_dimension, cnn_depth, l2_reg_weight, entropy_reg_weight: 5 x 2 x 2 x 4 = 80,
        window_size, indices, num_codes, feature_dimension, cnn_depth, entropy_reg_weight, total = 2 x 3 x 8 x 5 x 2 x 3 = 

    Args:
        model_parent_folder (str): folder containing child folders, each of which contains `config.json`, `dates.csv`, `features.npy`, `losses.png`, \
            multiple `model - {epochs}.pth`, `model_args.json`, `model_overview.txt`, and `training_log.json`.
    """
    feature_dimensions = [10, 20, 50, 100, 1000]
    cnn_depths = [4, 5]
    l2_reg_weights = [0, 0.1]
    entropy_reg_weights = [0.0, 0.5, 1.0, 2.0]
    indices = ['MXWO Index', 'VIX Index', 'CSI BARC Index', 'USGG10YR Index', 'DXY Curncy']

    # tune hyperperameters of feature_dimension, cnn_depth, l2_reg_weight, entropy_reg_weight,
    # with window_size = 128, slide_step = 10, images_encoding_method = gasf, lag = 1, len(indices) = 5, batch_size = 16,
    # num_codes = 6, learning_rate = 1e-5, num_epochs = 1
    for feature_dimension in feature_dimensions:
        for cnn_depth in cnn_depths:
            for l2_reg_weight in l2_reg_weights:
                for entropy_reg_weight in entropy_reg_weights:
                    current_time_string = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')
                    subprocess.run([
                        PYTHON_INTER,
                        'flow/end_to_end_flow.py',
                        '--num_codes', '6',
                        '--indices', ','.join(indices),
                        '--feature_dimension', f'{feature_dimension}',
                        '--model_folder', os.path.join(model_parent_folder, current_time_string),
                        '--results_folder', os.path.join(results_parent_folder, current_time_string),
                        '--l2_reg_weight', f'{l2_reg_weight}',
                        '--entropy_reg_weight', f'{entropy_reg_weight}',
                        '--cnn_depth', f'{cnn_depth}',
                        '--apply_norm'
                    ])

    window_sizes = [64, 128]
    indices_list = [
        ['MXWO Index', 'VIX Index', 'CSI BARC Index'],
        ['MXWO Index', 'VIX Index', 'CSI BARC Index', 'USGG10YR Index', 'DXY Curncy'],
        ''
    ]
    entropy_reg_weights = [0.5, 1.0, 2.0]
    num_codes_list = [3, 4, 5, 6, 7, 8, 9, 10]
    feature_dimensions = [10, 20, 50, 100, 1000]

    # tune hyperperameters of window_size, indices, num_codes, feature_dimension, cnn_depth, entropy_reg_weight,
    # with slide_step = 10, images_encoding_method = gasf, lag = 1, batch_size = 16, learning_rate = 1e-5, num_epochs = 1, l2_reg_weight = 0
    for window_size in window_sizes:
        for indices in indices_list:
            for num_codes in num_codes_list:
                for feature_dimension in feature_dimensions:
                    for cnn_depth in cnn_depths:
                        for entropy_reg_weight in entropy_reg_weights:
                            # skip num_codes == 6, len(indices) == 5, window size == 128, since these hyperparameter sets have been executed before
                            if num_codes == 6 and indices == ['MXWO Index', 'VIX Index', 'CSI BARC Index', 'USGG10YR Index', 'DXY Curncy'] and window_size == 128:
                                continue
                            current_time_string = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')
                            subprocess.run([
                                PYTHON_INTER,
                                'flow/end_to_end_flow.py',
                                '--window_size', f'{window_size}',
                                '--indices', ','.join(indices),
                                '--num_codes', f'{num_codes}',
                                '--feature_dimension', f'{feature_dimension}',
                                '--cnn_depth', f'{cnn_depth}',
                                '--entropy_reg_weight', f'{entropy_reg_weight}',
                                '--model_folder', os.path.join(model_parent_folder, current_time_string),
                                '--results_folder', os.path.join(results_parent_folder, current_time_string),
                                '--apply_norm'
                            ])


if __name__ == '__main__':
    os.environ['PYTHONPATH'] = os.path.abspath(os.path.dirname(__file__))

    ################### two-stage models ###################

    # experiment one, feature extraction based on correlation matrices -> cluster
    execute_similarity_generate_flow('data/inter data/correlation')
    execute_cluster_flow('data/inter data/correlation', add_hierarchical=True)

    # experiment two, feature extraction based on cnn auto encoder -> cluster
    execute_train_flow('data/inter data/deep learning')
    execute_cluster_flow('data/inter data/deep learning', add_hierarchical=False)

    # combination of one and two experiments, feature concatenation -> cluster
    deep_model_dir = utils.filter_deep_model_dir(
        'data/inter data/deep learning',
        window_size=128,
        slide_step=10,
        num_indices=5,
        learning_rate=1e-5,
        feature_dimension=20,
    )[0]
    correlation_dir = utils.filter_correlation_dir(
        'data/inter data/correlation',
        window_size=128,
        slide_step=10,
        num_indices=5,
        sim_method='meta',
        num_correlation_methods=3
    )[0]
    execute_feature_concat_flow(deep_model_dir, correlation_dir, 'data/inter data/concat')
    execute_cluster_flow('data/inter data/concat', add_hierarchical=False)

    ################### end-to-end model ###################
    # experiment three, Siamses cnn
    execute_end_to_end_flow('data/inter data/end-to-end', 'results/end-to-end')

