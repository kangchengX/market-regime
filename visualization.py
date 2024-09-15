import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from typing import List, Literal
from matplotlib.colors import CenteredNorm, Normalize
from matplotlib.colors import LinearSegmentedColormap
from constant import CUSTOM_PALETTE


def visualize_similarity_matrix(
        dates: pd.DataFrame,
        sim: np.ndarray,
        title: Literal['meta','cophenetic'], 
        figtitle: str | None = None,
        filename: str | None = None, 
        show = True, 
        figsize = (6,6), 
        formatter=r'%Y-%m-%d', 
        interval=1000,
        backend: str | None = None
):
    if backend is not None:
        matplotlib.use(backend)
    dates =  dates['DATE']
    fig, ax = plt.subplots(1,1,figsize=figsize)
    date_locator = mdates.DayLocator(interval=interval)

    # for ax, corr_matrix, title in zip(axs,corr_matices, titles):
    img = ax.imshow(sim, norm=CenteredNorm(halfrange=1),cmap='seismic')
    ax.set_title(title)
    ax.set_xticks(np.arange(len(dates)))
    ax.set_xticklabels(dates.dt.strftime(formatter),rotation=40)
    ax.xaxis.set_major_locator(date_locator)

    ax.set_yticks(np.arange(len(dates)))
    ax.set_yticklabels(dates.dt.strftime(formatter))
    ax.yaxis.set_major_locator(date_locator)
    fig.colorbar(img, ax=ax)
        
    if figtitle is not None:
        fig.suptitle(figtitle)

    if show:
        plt.show()

    if filename is not None:
        fig.savefig(filename, dpi=600)


def visualize_regimes(
        regimes_df: pd.DataFrame,
        fig_title: str | None='regimes',
        filename: str | None=None,
        show=True, 
        figsize=(18, 6), 
        formatter=r'%Y-%m-%d', 
        interval=1000,
        backend: str | None = None,
        vix: pd.DataFrame | None = None,
        align: bool | None = False
):
    '''
    Visualize the regime identification results:

    Args:
        regimes_df: dataframe that must include columns ('DATE', 'regime')
    '''
    
    regimes_df['regime'] = regimes_df['regime'] - regimes_df['regime'].min()

    if backend is not None:
        matplotlib.use(backend)
    date_locator = mdates.DayLocator(interval=interval)
    fig, ax = plt.subplots(figsize=figsize)

    if vix is not None:
        ax2 = ax.twinx()
        vix = pd.merge(vix, regimes_df, on = 'DATE')
        ax2.plot(regimes_df['DATE'], vix['VIX Index'])
    
    for regime, single_regime_df in regimes_df.groupby('regime'):
        ax.scatter(single_regime_df['DATE'], [1]*len(single_regime_df['regime']) if align else single_regime_df['regime'], color=plt.cm._colormaps.get_cmap('tab20')(int(regime) + 1), label = regime, s=1)
    
    ax.patch.set_visible(False)
    
    ax.set_yticks([0, 1, 2, 3, 4] if align else np.unique(regimes_df['regime']))
    ax.set_zorder(3)
    ax.set_yticklabels([]) if align else ax.set_ylabel('regime')

    fig.suptitle(fig_title)

    # ax.set_xticks(number_range)
    ax.set_xticks(regimes_df['DATE'].dt.strftime(formatter))
    ax.xaxis.set_major_locator(date_locator)
    ax.legend()


    ax.set_xlabel('date')

    if show:
        plt.show()

    if filename is not None:
        fig.savefig(filename)


def visualize_regime_durations(
        regime_durations_stats: pd.DataFrame, 
        filename: str | None = None,
        show: bool | None = True, 
        figsize: tuple | None = (12, 6), 
        fig_title: str | None = 'regime durations',
        y_lim: tuple | None = None,
        backend: str | None = None
):
    if backend is not None:
        matplotlib.use(backend)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize = figsize)

    axs[0].bar(
        regime_durations_stats.index, 
        regime_durations_stats['75 quantile'] - regime_durations_stats['25 quantile'], 
        bottom=regime_durations_stats['25 quantile'],
        width=0.4, 
        color='skyblue',
        alpha=0.6
    )
        
    axs[0].scatter(regime_durations_stats.index, regime_durations_stats['median'], label='Median', zorder=5)
    axs[0].scatter(regime_durations_stats.index, regime_durations_stats['mean'], label='Mean', zorder=5)
    axs[0].set_xticks(ticks = regime_durations_stats.index, labels = regime_durations_stats['regime'])

    if y_lim is not None:
        axs[0].set_ylim(y_lim[0],y_lim[1])
    else:
        axs[0].set_ylim(0, None)

    axs[0].legend()

    handles, _ = axs[0].get_legend_handles_labels()
    bar_patch = Patch(color='skyblue', label='25 to 75 percentile')
    handles.append(bar_patch)
    axs[0].legend(handles=handles)
    axs[0].set_title('durations for each period')

    axs[0].set_xlabel('regime')
    axs[0].set_ylabel('nuber of days')


    regime_durations_stats = regime_durations_stats[regime_durations_stats['regime'] != 'all'].reindex()
    axs[1].bar(
        regime_durations_stats.index, 
        regime_durations_stats['percent'], 
        width=0.4, 
        color='red',
        alpha=0.6
    )

    axs[1].set_ylim(0, 1)
    axs[1].set_xticks(ticks = regime_durations_stats.index, labels = regime_durations_stats['regime'])
    axs[1].set_title('durations for each regime in total')

    axs[1].set_xlabel('regime')
    axs[1].set_ylabel('percent')

    fig.suptitle(fig_title)
    
    if show:
        plt.show()

    if filename is not None:
        fig.savefig(filename)


def visualize_transition_matrix(
        transition_matrix: pd.DataFrame,
        filename: str | None = None,
        show: bool | None = True, 
        figsize: tuple | None = (6, 6), 
        fig_title: str | None = 'transition matrix',
        backend: str | None = None
):
    if backend is not None:
        matplotlib.use(backend)
    fig, ax = plt.subplots(figsize=figsize)

    img = ax.imshow(transition_matrix, norm=Normalize(0, 1), cmap='Reds')
    ax.set_xticks(transition_matrix.index)
    ax.set_xlabel('regime - to')
    ax.set_yticks(transition_matrix.index)
    ax.set_ylabel('regime - from')
    fig.colorbar(img, ax=ax)

    fig.suptitle(fig_title)

    if show:
        plt.show()

    if filename is not None:
        fig.savefig(filename)
        

def visualize_transition_matrix_sns(
        transition_matrix: pd.DataFrame,
        filename: str | None = None,
        show: bool | None = True, 
        figsize: tuple | None = (6, 5), 
        fig_title: str | None = 'transition matrix',
        annot: bool = True,
        fmt: str = ".2f",
        cbar: bool = True,
        backend: str | None = None
):
    if backend is not None:
        matplotlib.use(backend)

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#FFFFFF', '#FEE0D2', '#FCBBA1', '#FC9272', '#FB6A4A', '#EF3B2C', '#CB181D', '#A50F15', '#67000D']
    n_bins = 256  # Number of color gradations
    cmap_name = 'custom_reds'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    sns.heatmap(
        transition_matrix, 
        ax=ax, 
        annot=annot, 
        fmt=fmt, 
        cmap=cmap,
        cbar=cbar,
        vmin=0, 
        vmax=1, 
        linewidths=.5
    )
    ax.set_xlabel('regime - to')
    ax.set_ylabel('regime - from')
    fig.suptitle(fig_title)

    if show:
        plt.show()

    if filename is not None:
        fig.savefig(filename)


def visualize_losses(
        losses: list,
        figsize: tuple | None = (6,6),
        filename: str | None = None,
        show_fig: bool | None = True,
        backend: str | None = None
):
    """Visualize the losses against epochs
    
    Args:
        losses : list of losses for each epoch
        fig_size : tuple of 2 elements (height, width)
        filename : filename to save the plot. If None, the plot will not be saved. Default is None
        show_fig : if show the plot. True indicates show the plot
    """
    if backend is not None:
        matplotlib.use(backend)
    fig, ax = plt.subplots(figsize = figsize)
    ax.plot(range(1, len(losses)+1), losses)
    ax.set_ylim([0, 1.2])

    if filename is not None:
        fig.savefig(filename)
    
    if show_fig:
        plt.show()


def visualize_cluster_assess_on_returns(
        df: pd.DataFrame,
        fig_title: str,
        fig_size: tuple,
        show_fig: bool | None = True,
        filename: str | None = None,
        backend: str | None = None
):
    
    if backend is not None:
        matplotlib.use(backend)

    fig, ax = plt.subplots(figsize=fig_size)

    df_numeric = df.drop(columns='index')

    # Create a mask for NaN values to keep the text column space empty in the heatmap
    mask = np.column_stack([df[col].apply(lambda x: isinstance(x, (int, float))) for col in df])

    # Create a heatmap, but mask text columns so they don't get color coded
    sns.heatmap(df_numeric, annot=True, fmt=".3f", cmap="bwr", vmin=-1, vmax=1, mask=~mask[:, 1:], ax=ax)
    ax.set_xticks(np.arange(0.5, len(df_numeric.columns)))
    ax.set_xticklabels(df_numeric.columns)
    ax.set_yticks(np.arange(0.5, len(df)))
    ax.set_yticklabels(df['index'])

    # Rotate y-axis labels horizontally
    ax.tick_params(axis='y', rotation=0)

    fig.suptitle(fig_title)

    if filename is not None:
        fig.savefig(filename)
    
    if show_fig:
        plt.show()


def visualize_metrics_heatmaps(df: pd.DataFrame, metrics, filename: None = None):
    """
    Visualizes the heatmaps for the given metrics in the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    metrics (list of str): A list of metric names to visualize. The list should contain exactly three metric names.

    Returns:
    None: Displays the heatmaps.
    """
    if len(metrics) != 3:
        raise ValueError("The metrics list should contain exactly three metric names.")
    
    # Set up the matplotlib figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Loop through each metric and plot on corresponding axes
    for i, metric in enumerate(metrics):
        # Pivot the data for the current metric
        metric_pivot = df.pivot(index="regime", columns="index", values=metric).sort_index(ascending=False)
        
        # Plot heatmap on the corresponding subplot axis
        sns.heatmap(metric_pivot, annot=True, cmap='seismic' if i != 1 else 'seismic_r', center=0, ax=axes[i])
        axes[i].set_title(f'{metric}')
        axes[i].set_xlabel('')  # Remove the x-axis label
        axes[i].set_ylabel('')  # Remove the y-axis label

    # Adjust layout
    plt.tight_layout()

    # Save the figure if filename is provided
    if filename is not None:
        plt.savefig(filename)

    plt.show()


def visualize_sc_heatmap(
        df: pd.DataFrame,
        text_columns: List[str],
        numeric_columns: List[str],
        fig_title: str,
        fig_size: tuple,
        show_fig: bool | None = True,
        filename: str | None = None,
        backend: str | None = None
):
    if backend is not None:
        matplotlib.use(backend)

    df = df.sort_values(by=['regime', 'index'], ascending=[False,True])
    # Create a mask for NaN values to keep the text column space empty in the heatmap
    mask = np.column_stack([df[col].apply(lambda x: isinstance(x, (int, float))) for col in df])

    fig, ax = plt.subplots(figsize=fig_size)  # Use subplots for better control
    # Create a heatmap, but mask text columns so they don't get color coded
    # sns.heatmap(df[numeric_columns], annot=True, fmt=".3f", cmap="bwr", center=0, mask=~mask[:, len(text_columns):], ax=ax)
    df_numeric_normalized = df[numeric_columns].apply(lambda x : x / x.abs().max(), axis=0)
    df['average_return'] = df['average_return'].apply(lambda x : f'{x*100:.2f}%')
    df['std'] = df['std'].apply(lambda x : f'{x*100:.2f}%')
    df['ann_return'] = df['ann_return'].apply(lambda x : f'{x*100:.2f}%')
    df['ann_std'] = df['ann_std'].apply(lambda x : f'{x*100:.2f}%')
    df['sharpe_ratio'] = df['sharpe_ratio'].apply(lambda x : f'{x:.2f}')
    
    # hm = sns.heatmap(df_numeric_normalized, annot=df[numeric_columns], fmt=".4f", cmap="bwr", center=0, mask=~mask[:, len(text_columns):], ax=ax, cbar=False)
    hm = sns.heatmap(df_numeric_normalized, annot=df[numeric_columns], fmt='', cmap="bwr", center=0, mask=~mask[:, len(text_columns):], ax=ax, cbar=False)
    ax.set_xticks(np.arange(0.5, len(numeric_columns)))
    ax.set_xticklabels(numeric_columns)
    ax.set_yticks(np.arange(0.5, len(df)))
    if len(text_columns) == 2:
        ax.set_yticklabels([f"{' '.join(row[text_columns[1]].split()[:-1])} ({int(row[text_columns[0]])})" for index, row in df.iterrows()])
    else:
        ax.set_yticklabels(df[text_columns[0]])

    for i in range(8, len(df), 8):
        hm.hlines(i, *hm.get_xlim(), colors='black')

    # Rotate y-axis labels horizontally
    ax.tick_params(axis='y', rotation=0)

    fig.suptitle(fig_title)

    if filename is not None:
        fig.savefig(filename)
    
    if show_fig:
        plt.show()


def face_grid_5_dim(
    df: pd.DataFrame,
    metric: str,
    y_lim: tuple | None = None,
    save_name: str | None = None,
    grid_col: str | None = "window",
    grid_row: str | None = "indices",
    inner_x: str | None = 'clusters',
    inner_hue: str | None = 'pca',
    inner_style: str | None ='correlations',
):
    """
    Create a face grid for 5 hyperparameter dimensions for `metric`.

    Args:
        df (DataFrame): contains 5 hyperparameter columns  and 1 metric column.
        metric (str): metric column name of `df` for the y azis of the inner figure. 
        y_lim (tuple | None): tuple (y_min, y_max) for the y axis, i.e., vales of the metric. `None` indicates no limitaion. Default to `None`.
        save_name (str | None): path to save the figure. If `None`, the figure will not be saved. Default to `None`.
        grid_col (str): column name of `df` for the column of the grid. Default to `"window"`.
        grid_row (str): column name of `df` for the row of the grid. Default to `"indices"`.
        inner_x (str): column name of `df` for the x axis of the inner figure. Default to `'clusters'`.
        inner_hue (str): column name of `df` for hue of the inner figure, i.e. color of the plot. Default to `'pca'`.
        inner_style (str): column name of `df` for style of the inner figure, i.e. style of the plot. Default to `'correlations'`.
    """
    g = sns.FacetGrid(df, col=grid_col, row=grid_row, ylim=y_lim)
    # Map a scatterplot onto the grid
    g.map_dataframe(
        sns.lineplot,
        data=df,
        x=inner_x,
        y=metric,
        hue=inner_hue,
        style=inner_style,
        markers=True,
        dashes=True,
        errorbar=None,
        palette=CUSTOM_PALETTE
    )

    handles, labels = g.axes.flat[1].get_legend_handles_labels()
    g.axes.flat[1].legend(handles=handles, labels=labels)

    if save_name is not None:
        g.savefig(save_name)


def face_grid_4_dim(
    df: pd.DataFrame,
    metric: str,
    y_lim: tuple | None = None,
    save_name: str | None = None,
    grid_col: str | None = "window",
    grid_row: str | None = "indices",
    inner_x: str | None = 'clusters',
    inner_hue: str | None = 'pca'
):
    """
    Create a face grid for 5 hyperparameter dimensions for `metric`.

    Args:
        df (DataFrame): contains 5 hyperparameter columns  and 1 metric column.
        metric (str): metric column name of `df` for the y azis of the inner figure. 
        y_lim (tuple | None): tuple (y_min, y_max) for the y axis, i.e., vales of the metric. `None` indicates no limitaion. Default to `None`.
        save_name (str | None): path to save the figure. If `None`, the figure will not be saved. Default to `None`.
        grid_col (str): column name of `df` for the column of the grid. Default to `"window"`.
        grid_row (str): column name of `df` for the row of the grid. Default to `"indices"`.
        inner_x (str): column name of `df` for the x axis of the inner figure. Default to `'clusters'`.
        inner_hue (str): column name of `df` for hue of the inner figure, i.e. color of the plot. Default to `'pca'`.
    """
    g = sns.FacetGrid(df, col=grid_col, row=grid_row, ylim=y_lim)
    # Map a scatterplot onto the grid
    g.map_dataframe(
        sns.lineplot,
        data=df,
        x=inner_x,
        y=metric,
        hue=inner_hue,
        markers=True,
        dashes=True,
        errorbar=None,
        palette=CUSTOM_PALETTE,
        marker='o'

    )

    handles, labels = g.axes.flat[1].get_legend_handles_labels()
    g.axes.flat[1].legend(handles=handles, labels=labels)

    if save_name is not None:
        g.savefig(save_name)

    plt.close()

def face_grid_3_dim(
    df: pd.DataFrame,
    metric: str,
    y_lim: tuple | None = None,
    save_name: str | None = None,
    grid_col: str | None = "features",
    inner_x: str | None = 'clusters',
    inner_hue: str | None = 'pca'
):
    """
    Create a face grid for 5 hyperparameter dimensions for `metric`.

    Args:
        df (DataFrame): contains 5 hyperparameter columns  and 1 metric column.
        metric (str): metric column name of `df` for the y azis of the inner figure. 
        y_lim (tuple | None): tuple (y_min, y_max) for the y axis, i.e., vales of the metric. `None` indicates no limitaion. Default to `None`.
        save_name (str | None): path to save the figure. If `None`, the figure will not be saved. Default to `None`.
        grid_col (str): column name of `df` for the column of the grid. Default to `"window"`.
        grid_row (str): column name of `df` for the row of the grid. Default to `"indices"`.
        inner_x (str): column name of `df` for the x axis of the inner figure. Default to `'clusters'`.
        inner_hue (str): column name of `df` for hue of the inner figure, i.e. color of the plot. Default to `'pca'`.
    """
    g = sns.FacetGrid(df, col=grid_col, ylim=y_lim)
    # Map a scatterplot onto the grid
    g.map_dataframe(
        sns.lineplot,
        data=df,
        x=inner_x,
        y=metric,
        hue=inner_hue,
        markers=True,
        dashes=True,
        errorbar=None,
        palette=CUSTOM_PALETTE,
        marker='o'
    )

    handles, labels = g.axes.flat[1].get_legend_handles_labels()
    g.axes.flat[1].legend(handles=handles, labels=labels)

    if save_name is not None:
        g.savefig(save_name)

    plt.close()


def visualize_summary_per_hyperparameter(
        df: pd.DataFrame, 
        metric: str, 
        color_order: int | None, 
        hyperparameters = ['sim_method', 'correlations', 'window', 'indices', 'pca'],
        filename=None,
        show=False,
):
    fig, axes = plt.subplots(1, len(hyperparameters), figsize=(4 * len(hyperparameters), 4))  # 2x2 grid for 4 hyperparameters

    # adjust y_lim
    if metric == 'clusters' or metric == 'match_codes':
        count_max = 0

    # Plot each hyperparameter in a separate subplot
    for ax, hyperparameter in zip(axes, hyperparameters):
        if metric == 'clusters' or metric == 'match_codes':
            sns.countplot(x=hyperparameter, hue=metric, data=df, ax=ax)
            count_max = max(count_max, df.groupby([metric, hyperparameter]).count().max().max())
            ax.set_ylabel('count')
        else:
            sns.boxplot(x=hyperparameter, y=metric, data=df, ax=ax, color = sns.color_palette()[color_order])
            ax.set_ylabel(metric)

        ax.set_xlabel(hyperparameter)
    
    if metric == 'clusters' or metric == 'match_codes':
        for ax in axes:
            ax.set_ylim(0, count_max + 2)

    # Adjust layout
    plt.tight_layout()
    if filename is not None:
        fig.savefig(filename)
    if show:
        plt.show()
    plt.close()


def visualize_summary_for_corr_not_optimal(
    df: pd.DataFrame, 
    fix_column_value: tuple, 
    metric: str,
    grid_col="window",
    grid_row="indices",
    inner_x='clusters',
    inner_hue='pca',
    inner_style='correlations'
):
    df = df[
        (~df['find_optimal'])
        &(df[fix_column_value[0]]==fix_column_value[1])
    ].drop(columns=['find_optimal', fix_column_value[0]])
    max_metirc = df[metric].max()
    min_metric = df[metric].min()
    y_lim = [min_metric, max_metirc]
    df_meta = df[df['sim_method']=='meta'].drop(columns=['sim_method'])
    df_cophenetic = df[df['sim_method'] == 'cophenetic'].drop(columns=['sim_method'])
    face_grid_5_dim(
        df=df_meta, 
        y_lim=y_lim, 
        save_name=fix_column_value[0]+'fixed-'+metric+'-meta', 
        metric=metric,
        grid_col=grid_col,
        grid_row=grid_row,
        inner_x=inner_x,
        inner_hue=inner_hue,
        inner_style=inner_style
    )
    face_grid_5_dim(
        df=df_cophenetic, 
        y_lim=y_lim, 
        save_name=fix_column_value[0]+'fixed-'+metric+'-cophenetic', 
        metric=metric,
        grid_col=grid_col,
        grid_row=grid_row,
        inner_x=inner_x,
        inner_hue=inner_hue,
        inner_style=inner_style
    )


def visualize_summary_for_corr_optimal(
        df: pd.DataFrame,
        metric: str,
        color_order: int | None
):
    df = df[(df['find_optimal']) & (df['slide'] == 10)].drop(columns=['find_optimal', 'slide'])
    visualize_summary_per_hyperparameter(
        df=df, 
        metric=metric,
        color_order=color_order, 
        filename=metric+'-optimal',
        hyperparameters=['sim_method', 'correlations', 'window', 'indices', 'pca']
    )

def visualize_summary_for_deep_not_optimal(
    df: pd.DataFrame, 
    metric: str,
    grid_col="indices",
    grid_row="window",
    inner_x='clusters',
    inner_hue='pca'
):
    df_fix_features = df[
        (~df['find_optimal'])
        &(df['features']==20)
    ].drop(columns=['find_optimal', 'features'])

    df_only_features = df[
        (~df['find_optimal'])
        &(df['window'] == 128)
        &(df['indices'] == 5)
    ].drop(columns=['find_optimal', 'window', 'indices'])

    y_lim_fix_features = [df_fix_features[metric].min(), df_fix_features[metric].max()]
    y_lim_only_features = [df_only_features[metric].min(), df_only_features[metric].max()]

    face_grid_4_dim(
        df=df_fix_features, 
        y_lim=y_lim_fix_features, 
        save_name='features'+'-fixed-'+metric, 
        metric=metric,
        grid_col=grid_col,
        grid_row=grid_row,
        inner_x=inner_x,
        inner_hue=inner_hue
    )

    face_grid_3_dim(
        df=df_only_features, 
        y_lim=y_lim_only_features, 
        save_name='features'+'-vary-'+metric, 
        metric=metric,
        grid_col='features',
        inner_x=inner_x
    )


def visualize_summary_for_deep_optimal(
        df: pd.DataFrame,
        metric:  str,
        color_order: int | None
):
    df = df[
        (df['find_optimal'])
        &(df['features'] == 20)
    ].drop(columns=['find_optimal', 'features'])
    visualize_summary_per_hyperparameter(
        df=df,
        metric=metric,
        color_order=color_order,
        filename=metric+'-optimal',
        hyperparameters=['window', 'indices', 'pca']
    )


def visualize_summary_for_concat_not_optimal(
        df: pd.DataFrame,
        metric: str,
):
    df = df[~df['find_optimal']].drop(columns='find_optimal')
    y_lim = [df[metric].min(), df[metric].max()]
    face_grid_3_dim(
        df=df, 
        metric=metric,
        y_lim=y_lim,
        save_name=metric,
        grid_col='weight_corr',
    )


def visualize_summary_for_concat_optimal(
        df: pd.DataFrame,
        metric: str,
        color_order: int | None
):
    df = df[df['find_optimal']].drop(columns='find_optimal')
    visualize_summary_per_hyperparameter(
        df=df,
        metric=metric,
        color_order=color_order,
        filename=metric+'-optimal',
        hyperparameters=['weight_corr', 'pca']
    )


def visualize_summary_for_end_to_end(
        df: pd.DataFrame,
        metric: str,
        color_order: int | None
):
    df = df[(df['l2_reg'] == 0) & (df['entropy_reg'] != 0)].drop(columns='l2_reg')
    df['match_codes'] = df['clusters_real'] == df['num_codes']
    df['match_codes'] = df['match_codes'].apply(lambda x : 'match' if x else 'not match')
    if metric != 'match_codes':
        df=df[df['match_codes']=='match']
    else:
        # add column 'match_codes'
        df['match_codes'] = df['clusters_real'] == df['num_codes']
        df['match_codes'] = df['match_codes'].apply(lambda x : 'match' if x else 'not match')
    visualize_summary_per_hyperparameter(
        df=df,
        metric=metric,
        hyperparameters=['window', 'indices', 'features', 'cnn_depth', 'entropy_reg', 'num_codes'],
        filename=metric,
        color_order=color_order,
    )
