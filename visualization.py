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

        
# def visualize_regimes(
#         corr_dates:pd.Series,
#         labels_all:List[np.ndarray],
#         titles:List[str]|None=['regime_cophenetic','regime_meta'], 
#         fig_title:str | None='regimes',
#         filename:str|None=None,
#         show=True, 
#         figsize=(12,6), 
#         nrows=1,
#         ncols=2,
#         formatter=r'%Y-%m-%d', 
#         interval=1000
# ):
#     '''Visualize the regime identification results:

#     Args:
#         corr_dates
#     '''
    
#     assert len(labels_all) == len(titles) == nrows*ncols

#     number_range = np.arange(len(corr_dates))
#     date_locator = mdates.DayLocator(interval=interval)
#     fig, axs = plt.subplots(nrows,ncols,figsize=figsize)

#     if len(labels_all) == 1:
#         axs = [axs]

#     for labels, ax, title in zip(labels_all,axs, titles):

#         for label in np.unique(labels):
#             subset = labels == label
#             ax.scatter(number_range[subset],labels[subset],color=plt.cm._colormaps.get_cmap('Set1')(int(label)), label = label, s=1)

#         ax.set_xticks(np.arange(len(corr_dates)))
#         ax.set_xticklabels(corr_dates.dt.strftime(formatter),rotation=40)
#         ax.xaxis.set_major_locator(date_locator)
#         ax.set_title(title)
#         ax.legend()

#         ax.set_yticks(np.unique(labels))

#     fig.suptitle(fig_title)

#     if show:
#         plt.show()

#     if filename is not None:
#         fig.savefig(filename)


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
    '''Visualize the regime identification results:

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

        # for regime, single_regime_df in regimes_df.groupby('regime'):
        #     ax2 = ax.twinx()
        #     ax2.scatter(single_regime_df['DATE'], [1]*len(single_regime_df['regime']) if align else single_regime_df['regime'], color=plt.cm._colormaps.get_cmap('tab20')(int(regime)), label = regime, s=1)
        # ax2.set_yticks([0, 1, 2, 3, 4] if align else np.unique(regimes_df['regime']))
        # ax2.set_yticklabels([]) if align else ax2.set_ylabel('regime')
    
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
    # num_regimes = len(regime_durations_stats) - 1
    # labels = regime_durations_stats['regime']
    # regime_durations_stats[regime_durations_stats['regime'] == 'all']['regime'] = num_regimes
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


# def visualize_regime_durations(
#         regime_durations_stats_all: List[pd.DataFrame], 
#         titles: List[str]| None = [None], 
#         filename: str | None = None,
#         show: bool | None = True, 
#         figsize: tuple | None = (6,6), 
#         nrows: int | None = 1,
#         ncols: int | None = 1,
#         fig_title: str | None = 'regime durations',
#         y_lim: tuple | None = None
# ):
#     assert len(regime_durations_stats_all) == len(titles) == nrows*ncols

#     fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize = figsize)

#     max_duration = np.max([durations.drop(columns='std').max().max() for durations in regime_durations_stats_all])

#     if len(regime_durations_stats_all) == 1:
#         axs = [axs]

#     for regime_durations, ax, title in zip(regime_durations_stats_all, axs, titles):
#         regime_durations = regime_durations.drop('all', errors='ignore')
#         for regime, single_regime_duration in regime_durations.iterrows():
#             ax.bar(regime, 
#                 single_regime_duration['75 quantile'] - single_regime_duration['25 quantile'], 
#                 bottom=single_regime_duration['25 quantile'],
#                 width=0.4, 
#                 color = 'skyblue',
#                 alpha=0.6)
            
#         ax.scatter(regime_durations.index.to_list(), regime_durations['median'].to_list(),label='Median',zorder=5)
#         ax.scatter(regime_durations.index.to_list(), regime_durations['mean'].to_list(),label='Mean',zorder=5)
#         ax.set_xticks(regime_durations.index.to_list())

#         if y_lim is None:
#             ax.set_ylim(0, max_duration + 10)
#         else:
#             ax.set_ylim(y_lim[0],y_lim[1])

#         ax.legend()

#         handles, labels = ax.get_legend_handles_labels()
#         bar_patch = Patch(color='skyblue', label='25 to 75 percentile')
#         handles.append(bar_patch)
#         ax.legend(handles=handles)
#         if title is not None:
#             ax.set_title(title)

#     fig.suptitle(fig_title)
    
#     if show:
#         plt.show()

#     if filename is not None:
#         fig.savefig(filename)



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


def visualize_heatmap(
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

if __name__ == '__main__':
    import json
    # import stats
    # regimes = {
    #     'DATE': pd.date_range(start='2001-01-01', periods=1000, freq='D'),
    #     'regime': np.random.choice([0,1,2,3], 1000, replace=True),
    #     'price1': np.random.random(1000),
    #     'price2': np.random.random(1000)
    # }
    # regimes_df = pd.DataFrame(regimes)

    # stats_d = stats.analyse_regimes_durations(regimes_df)
    # transition_matrix = stats.calculate_transition_matrix_regime_level(regimes_df)
    # print(stats_d)
    # visualize_transition_matrix(transition_matrix)

    # with open(r'data\inter data\deep learning\2024-07-18 11-31-56\training_log.json', 'r') as f:
    #     config = json.load(f)
    #     losses = config['losses']

    # visualize_losses(losses=losses)

# def visualize_transition_matrix(
#         matrix: np.ndarray,
#         figsize: tuple | None = (6,6),
#         filename: str | None = None,
#         show_fig: bool | None = True):
#     normalizer = Normalize(0, 1, clip=True)
#     fig, axs = plt.subplots(figsize = figsize)
#     ax.imshow()
