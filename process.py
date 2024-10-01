import time, os, torch, json
import stats, cluster, visualization, utils, correlation
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from typing import Literal, List
from warnings import warn
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from data import TimeSeriesCorrelationsDataset
from loss import KLDivergenceLoss


class Processor():
    """The processor for model training and inference."""
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        data_loader: DataLoader,
        learning_rate: float,
        save_folder: str,
        optimizer_type: Literal['adam', 'sgd'] | None = 'adam',
    ):
        """
        Initialize the model.
        
        Args:
            model (Module): The model to train and inference.
            criterion (Module): The loss object.
            data_loader (DataLoader): the data loader containing the training data and the data for inference.
            learning_rate (float): The learning rate for model training.
            save_folder (str): Folder to save the results and model.
            optimizer_type (str): Type of the optimizer. Default to `'adam'`.
        """
        # set the model to cuda if not
        if not next(model.parameters()).is_cuda:
            self.model = model.to('cuda')
        else:
            self.model = model
        
        self.criterion = criterion
        self.data_loader = data_loader
        self.save_folder = save_folder

        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        if self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f'Unsuported optimizer_type : {optimizer_type}')
        
        self.losses = []
        self.epochs = 0


    def train(
        self, 
        epochs: int, 
        save_model: bool | None = True,
        save_period : int | None = None,
        save_log: bool | None = True,
        siamese: bool | None = False
    ):
        """
        Train ther model. If the model has been trained, the model will be trained based on the previous trained parameters.
        
        Args:
            epochs (int): Number of epochs for this training process.
            save_model (bool): If `True`, save the model with name self.directory + '.pth' in the folder self.parent_folder / self.directory. Default to `True`.
            save_period (int | None): If not `None`, save the model every save_period of epochs. Default to `None`.
            save_log (bool): If `True`, save 'batch_size', 'num_epochs', 'learning_rate' and 'losses' in a json file.
                named self.directory + '.json' in the folder self.parent_folder / self.directory. Default to `Trur`.
            siamese (bool): If `True`, the 'targets' from `self.data_loader` will be fed to the model. Default to `False`.

        Returns:
            out (list): List of the losses including those from the training process before.
        """
        self.model.train()
        for _ in range(epochs):
            t1 = time.time()
            batch_losses = []

            for inputs, targets in self.data_loader:
                inputs = inputs.to('cuda')
                targets = targets.to('cuda')
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                if siamese:
                    targets = self.model(targets)
                if isinstance(self.criterion, KLDivergenceLoss):
                    loss = self.criterion(outputs, targets, self.model.parameters())
                else:
                    loss = self.criterion(outputs, targets)
                batch_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()

            t2 = time.time()
            epoch_loss = np.mean(batch_losses)
            print(f"Epoch {self.epochs+1}, Loss : {epoch_loss}, Time : {t2-t1}")
            self.epochs += 1
            self.losses.append(epoch_loss)

            if save_period is not None and self.epochs % save_period == 0:
                torch.save(self.model, os.path.join(self.save_folder, f'model - {self.epochs}.pth'))

        if save_model:
            torch.save(self.model.state_dict(), os.path.join(self.save_folder, f'model - {self.epochs}.pth'))

        if save_log:
            self.save_log()

        return self.losses

    def save_log(self):
        """
        Save training log in a json file named self.save_folder/training_log.json.
        """
        loss_config = None
        if isinstance(self.criterion, nn.MSELoss):
            loss_type = 'mse'
        elif isinstance(self.criterion, nn.KLDivLoss):
            loss_type = 'kldiv'
        elif isinstance(self.criterion, KLDivergenceLoss):
            loss_type = 'kldiv'
            loss_config = {
                'lambda_weights' : self.criterion.l2_reg_weight,
                'lambda_entropy' : self.criterion.entropy_reg_weight
            }
        else:
            loss_type = None
            warn('This loss type cannot be interpreted to a string')
        training_info = {
            'batch_size' : self.data_loader.batch_size,
            'num_epochs' : self.epochs,
            'learning_rate' : self.learning_rate,
            'losses' : self.losses,
            'optimizer_type' : self.optimizer_type,
            'loss_type': loss_type,
            'loss_config': loss_config
        }
        with open(os.path.join(self.save_folder, 'training_log.json'), 'w') as f:
            json.dump(training_info, f, indent=4)

    
    def extract_feature_representation(self, save: bool | None = True):
        """
        Extract the features from the encoder. Only for the CNN AutoEncoder.
        
        Args:
            save (bool): If `True`, the array of features will be saved with path self.save_folder/features.npy, and dates will be saved
                with path self.save_folder/dates.csv if the passed dataloader's `date` attribute is not `None`. Default to `True`.

        Returns:
            features (ndarray): The extracted features.
            dates (DataFrame): DataFrame of the corresponding dates, returned if the passed dataloader's `date` attribute is not `None`.
        """
        encoder = self.model.encoder
        encoder.eval()
        features = np.concatenate(
            [encoder(inputs.to('cuda')).to('cpu').detach().numpy() 
             for inputs, _ in self.data_loader],
            0
        )
        
        if save:
            np.save(os.path.join(self.save_folder, 'features'), features)
            if self.data_loader.dataset.dates is not None:
                self.data_loader.dataset.dates.to_csv(os.path.join(self.save_folder, 'dates.csv'))

        if self.data_loader.dataset.dates is not None:
            return features, self.data_loader.dataset.dates

        else:
            return features
        
    
    def extract_regimes(self, save: bool | None = False) -> pd.DataFrame:
        """
        Extract regimes. Only for the Siamese CNN.

        Args:
            save (bool): If `True`, the extracted regimes will be saved to self.save_folder/regimes.csv.

        Returns:
            regimes_df (DataFrame). The dataframe of the extracted regimes.
        """
        self.model.eval()
        regimes = np.concatenate(
            [self.model(inputs.to('cuda')).to('cpu').detach().numpy() 
             for inputs, _ in self.data_loader],
            0
        )
        if self.data_loader.dataset.dates is None:
            raise RuntimeError('dataset need to have not None attribute dates')
        
        regimes_df = self.data_loader.dataset.dates.copy()
        regimes_df['regime'] = regimes

        if save:
            regimes_df.to_csv(os.path.join(self.save_folder, 'regimes.csv'))

        return regimes_df


class SimGenerator():
    """The class to generate similarity matrix."""
    def __init__(self, dataset: TimeSeriesCorrelationsDataset, method: Literal['meta','cophenetic'], save_folder: str):
        """
        Initialize the class.

        Args:
            dataset (TimeSeriesCorrelationsDataset): Dataset of the correlation matrices.
            method (str): The method to generate the similarity matrix.
            save_folder (int): Folder to save the similarity matrix.
        """
        self.dataset = dataset
        if method not in ['meta','cophenetic']:
            raise ValueError(f"Unsupported method {method}. This should be meta or cophenetic.")
        self.method = method
        self.save_folder = save_folder

    def generate_features(self, save: bool | None = True):
        """
        Generate the the similarity matrix, i.e., the features.

        Args:
            save (bool): If `True`, the similarity matrix will be saved to self.save_folder/dates.csv. Default to `True`.

        Returns:
            sim (ndarray): The generated similarity matrix.
        """
        if self.method == 'meta':
            sim = correlation.generate_meta_similarity(self.dataset.data)
        else:
            sim = correlation.generate_cophenetic_similarity(self.dataset.data)

        if save:
            np.save(os.path.join(self.save_folder, 'features'), sim)
            self.dataset.dates.to_csv(os.path.join(self.save_folder, 'dates.csv'))

        return sim


class Analyser:
    """The class to perform the analysis on the identified regimes."""
    def __init__(self, returns_df: pd.DataFrame, returns_dfs: List[pd.DataFrame]):
        """
        Initialize the class.

        Args:
            returns_df (DataFrame): The DataFrame with columns (`'DATE'`, ...) (the other columns are daily returns with the index name as the first row).
            returns_dfs (DataFrame): List of Dataframes, where each dataframe contains columns (`'DATE'`, ...) where other columns are trailing or forward returns.
        """
        self.returns_df = returns_df
        self.returns_dfs = returns_dfs

    def analyse(self, features: np.ndarray | None, original_regimes_df: pd.DataFrame, folder: str, return_filled_regimes: bool | None = True):
        """
        Perform the analysis.

        Args:
            original_regimes_df (DataFrame): The original regimes, i.e., those with sliding step. This has columns (`'DATEA'`, `'regimes'`).
            folder (str): Folder to save the analysis results.
            return_filled_regimes (bool): If `True`, the filled regimes will be returned.
        """
        # check if only one regime
        if len(original_regimes_df['regime'].unique()) == 1:
            print('no need to analyze since only one regime')
            if return_filled_regimes:
                regimes_df = self.returns_df[['DATE']].copy()
                regimes_df['regime'] = 0
                return regimes_df
            return

        returns_regimes_df = utils.fill_dates_values(
            df = self.returns_df, 
            regimes_df=original_regimes_df
        )
        returns_regimes_df = returns_regimes_df.sort_values(by='DATE')

        # durations
        regimes_durations_stats = stats.analyse_regimes_durations(returns_regimes_df)
        transition_matix_regime_level = stats.calculate_transition_matrix(returns_regimes_df, at_regimes_level=True)
        transition_matix_date_level = stats.calculate_transition_matrix(returns_regimes_df, at_regimes_level=False)
        
        transition_matix_regime_level_entropies = {
            'unnorm' : stats.calculate_transition_matrix_entropy(transition_matix_regime_level),
            'norm' : stats.calculate_transition_matrix_entropy(transition_matix_regime_level, apply_norm=True)
        }    

        transition_matix_date_level_entropies = {
            'unnorm' : stats.calculate_transition_matrix_entropy(transition_matix_date_level),
            'norm' : stats.calculate_transition_matrix_entropy(transition_matix_date_level, apply_norm=True)
        }
        
        # returns
        returns_stats_within_regimes = stats.calculate_return_metrics_within_regime(returns_regimes_df)
        returns_stats_last_dates = stats.calculate_return_metrics_last_date_of_period(returns_regimes_df)
        returns_forward_10days = stats.calculate_returns_forward(returns_regimes_df, 10)

        # cluster analysis
        if features is None:
            cluster_ana_results = pd.DataFrame()
        else:
            cluster_ana_results = cluster.assess_clustering_results(features, original_regimes_df)
        cluster_ana_on_returns = cluster.assess_clustering_on_returns(dfs=self.returns_dfs, regimes=returns_regimes_df)


        results = {
            'durations_ana' : {
                'stats' : regimes_durations_stats.to_dict(),
                'transition_matrix' : {
                    'regime_level' : {
                        'matrix' : transition_matix_regime_level.to_dict(),
                        'entropies' : transition_matix_regime_level_entropies
                    },
                    'data_level' : {
                        'matrix' : transition_matix_date_level.to_dict(),
                        'entropies': transition_matix_date_level_entropies
                    }
                }
            },
            'returns_ana' : {
                'within_regimes' : returns_stats_within_regimes.to_dict(),
                'last_dates' : returns_stats_last_dates.to_dict(),
                'forward_10days' : returns_forward_10days.to_dict()
            },
            'cluster_ana' : cluster_ana_results.to_dict(),
            'cluster_ana_on_returns': cluster_ana_on_returns.to_dict()
        }

        if folder is not None:

            with open(os.path.join(folder, 'analysis.json'), mode='w') as f:
                json.dump(results, f, cls=utils.NumpyEncoder, indent=8)

            with open(os.path.join(folder, 'analysis.csv'), 'w', newline='\n') as f:
                f.write('regimes durations stats' + '\n')
                regimes_durations_stats.to_csv(f, index=False)
                f.write('transition matix at regime level' + '\n')
                transition_matix_regime_level.to_csv(f)
                pd.Series(transition_matix_regime_level_entropies, name='entropy').to_csv(f)
                f.write('transition matix at date level' + '\n')
                transition_matix_date_level.to_csv(f)
                pd.Series(transition_matix_date_level_entropies, name='entropy').to_csv(f)
                f.write('returns stats within each regime for each index' + '\n')
                returns_stats_within_regimes.to_csv(f, index=False)
                f.write('returns stats for the last date of the period for each regime for each index' + '\n')
                returns_stats_last_dates.to_csv(f, index=False)
                f.write('returns stats for 10 days returns forward' + '\n')
                returns_forward_10days.to_csv(f, index=False)
                f.write('cluster assessment' + '\n')
                cluster_ana_results.to_csv(f)
                f.write('cluster assessment on returns' + '\n')
                cluster_ana_on_returns.to_csv(f, index=False)

            visualization.visualize_regime_durations(
                regimes_durations_stats,
                filename=os.path.join(folder, 'durations.png'),
                show=False,
                backend='Agg'
            )
        
            visualization.visualize_regimes(
                returns_regimes_df,
                filename=os.path.join(folder, 'regimes.png'),
                show=False,
                backend='Agg'
            )

            visualization.visualize_transition_matrix(
                transition_matix_regime_level,
                filename=os.path.join(folder, 'transition matrix ' + 'regime level.png'),
                show=False,
                backend='Agg'
            )

            visualization.visualize_transition_matrix(
                transition_matix_date_level,
                filename=os.path.join(folder, 'transition matrix ' + 'date level.png'),
                show=False,
                backend='Agg'
            )

            
            visualization.visualize_cluster_assess_on_returns(
                cluster_ana_on_returns[cluster_ana_on_returns['metric'] == 'silhouette'].drop(columns='metric'),
                fig_size=(12,4),
                fig_title='evaluation of cluster on returns',
                show_fig=False,
                filename=os.path.join(folder, 'cluster_on_returns.png'),
                backend='Agg'
            )

        if return_filled_regimes:
            return returns_regimes_df[['DATE','regime']]
