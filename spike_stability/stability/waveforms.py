from os import stat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import SubplotSpec
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pylab
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import gaussian_filter1d
import scipy.stats as stats
from itertools import combinations, product
from colorutils import Color

def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontweight='semibold')
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')
    
def waveforms_overtime(df_,
                    cluster_key:str,
                    date_key:str,
                    spike_cols:list,
                    color_dic:dict,
                    savefig:bool,
                    savefigpath='',
                    title=True,
                    borders=False):
    '''
    Function to plot average waveforms +- 1 st.dev over time for each neuron
    Args:
        df (pandas dataframe)
        cluster_key (str): cluster identifier, must be in df
        date_key (str): cluster identifier, must be in df
        color_dic (dict): mapping dict between neurons and colors
        savefig (bool)
        savefigpath (str)
        title (bool): whether to include title for each timepoint
        borders (bool)
    Returns:
    '''
    c = df_[cluster_key]
    d = df_[date_key]
    X = df_[spike_cols]
    title_dates = np.unique(d)
    nb_d, nb_c = len(np.unique(d)), len(np.unique(c))
    fig, axs = plt.subplots(nb_d, nb_c, figsize=(5*nb_c,5*nb_d), sharex=True, sharey=True)
    for i, timepoint in enumerate(sorted(np.unique(d), key=int, reverse=True)):
        mask_d = d == timepoint
        for j, clus in enumerate(np.unique(c)):
            mask_c = c == clus
            mask = mask_c & mask_d
            X_ = X[mask]
            t = np.arange(X_.shape[1])
            #for el in X_:
             #   axs[i, j].plot(t, el, c='grey', alpha=0.5)
            stdev = np.std(X_, axis=0)
            mean = np.mean(X_, axis=0)
            if nb_c > 1:
                axs[i, j].plot(t, mean, c=color_dic[str(clus)], lw=10.0)
                axs[i, j].fill_between(t, mean+stdev, mean-stdev, facecolor='grey', alpha=0.5)
            else:
                axs[i].plot(t, mean, c=color_dic[str(clus)], lw=10.0)
                axs[i].fill_between(t, mean+stdev, mean-stdev, facecolor='grey', alpha=0.5)
            if not borders:
                if nb_c > 1:
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
                    axs[i, j].axis("off")
                else:
                    axs[i].set_xticks([])
                    axs[i].set_yticks([])
                    axs[i].axis("off")
    if title:
        grid = plt.GridSpec(nb_d, nb_c)
        for i in range(int(nb_d)):
            create_subtitle(fig, grid[i, ::], 'Timepoint '+str(title_dates[-(i+1)]))
    if savefig:
        plt.savefig(savefigpath+'.jpg', format='jpg')
        plt.savefig(savefigpath+'.pdf', format='pdf')
    plt.show()
            
def all_channel_waveforms_overtime(df,
                                mouse:int,
                                cluster_key:str,
                                channel_key:str,
                                date_key:str,
                                spike_cols:list,
                                all_color_dics:dict,
                                savefig:bool,
                                color_key:str,
                                savefig_basepath='',
                                title=True,
                                borders=False):
    '''
    Function to plot average waveforms +- 1 st.dev over time for each neuron
    Args:
        df (pandas dataframe)
        mouse (int)
        cluster_key (str): cluster identifier, must be in df
        channel_key (str): channel identifier, must be in df
        all_color_dics (dict of dict): mapping dict between channelXXmouseY and {'waveclus':{"0": '#FFF000", ...}, ...}
        MAKE SURE that mouse and channel correspon with the colors you want in the all_color_dics dict.
        color_key (str): key fir dictionary in all_color_dics
        color_key (str): for color dict
        savefig (bool)
        savefig_basepath (str)
        title (bool): whether to include title for each timepoint
        borders (bool)
    Returns:
    '''
    df_ = df[df['mouse'] == mouse]
    for i, chan in enumerate(df_[channel_key].unique()):
        df_chan = df_[df_[channel_key] == chan]
        savefig_path_ = savefig_basepath + 'waveforms_over_time'+'channel_' + str(chan)+'mouse_'+str(mouse)+str(cluster_key)
        color_dic = all_color_dics['channel'+str(chan)+'mouse'+str(mouse)][color_key]
        waveforms_overtime(df_chan, cluster_key, date_key, spike_cols, color_dic, savefig, savefig_path_, title, borders)

def average_waveforms(cluster_id, cluster_labels,
                      X, stages):
    '''
    Comput average neuron waveforms for each timepoint.
    Args:
        cluster_id (str or int): cluster_id to select
        cluster_labels (list of str or int)
        X (2D array): spike waveforms
        stages (list of str or int): actual dates
    Returns:
        avg_wfs (list of array): list of mean waveforms for each date
    '''
    mask = cluster_labels == cluster_id
    wfs_this_cluster = X[mask]
    stage_labels_this_cluster = stages[mask]
    avg_wfs = [np.mean(wfs_this_cluster[stage_labels_this_cluster == day], axis=0) for day in np.unique(stage_labels_this_cluster)]
    return avg_wfs
        
def autocorrelation(avg_wfs, stat='pearsonr'):
    '''
    Calculate autocorrelation between average neuron waveforms of of one neuron across
    timepoints.
    Args:
        avg_wfs (list of arrays): output of average_waveforms
    Returns:
        corr_coefs (list): correlation coefficients
        p_values (list)
        date1_comparison (list)
        date2_comparison (list)
    '''
    corr_coefs = []
    p_values = []
    date1_comparison = []
    date2_comparison = []
    for (x, y), (i, j) in zip(list(combinations(avg_wfs, r=2)),
                          list(combinations(range(len(avg_wfs)), r=2))):
        if stat == 'pearsonr':
            r, p = stats.pearsonr(x, y)
        elif stat == 'spearmanr':
            r, p = stats.spearmanr(x, y)
        corr_coefs.append(r)
        p_values.append(p)
        date1_comparison.append(i+1)
        date2_comparison.append(j+1)
    return corr_coefs, p_values, date1_comparison, date2_comparison

def cross_correlations(avg_wfs_1, avg_wfs_2, stat='pearsonr'):
    '''
    Calculate cross correlation between two lists of average waveforms.
    Args:
        avg_wfs_1 (list of array): 
        avg_wfs_2 (list of array): 
    Returns:
        corr_coefs (list)
        p_values (list)
        dates (list)
    '''
    corr_coefs = []
    p_values = []
    dates = []
    for date, (x, y) in enumerate(zip(avg_wfs_1, avg_wfs_2)):
        if stat == 'pearsonr':
            r, p = stats.pearsonr(x, y)
        elif stat == 'spearmanr':
            r, p = stats.spearmanr(x, y)
        corr_coefs.append(r)
        p_values.append(p)
        dates.append(date+1)
    return corr_coefs, p_values, dates

def autocorr_histogram(cluster_labels,
                       X, stages, color_dic, ax,
                       min_corr=0.8, nb_bins=100, stat='pearsonr'):
    bins = np.linspace(min_corr, 1., nb_bins)
    for cluster_id in np.unique(cluster_labels):
        avg_wfs = average_waveforms(cluster_id, cluster_labels,
                                    X, stages)
        autocorr_coefs, p_values, dates1, dates2 = autocorrelation(avg_wfs)
        h, b = np.histogram(autocorr_coefs, bins=bins, density=False)
        h = (h/len(autocorr_coefs))*100
        ax.step(bins[:-1], h, color=color_dic[int(cluster_id)], lw=2.)
    if stat == 'pearsonr':
        ax.set_xlabel('Pearson correlation coefficient')
    elif stat == 'spearmanr':
        ax.set_xlabel('Spearman correlation coefficient')
    ax.set_ylabel('Percentage of total comparisons %')
    ax.legend(loc='upper left')
    return ax
        
def all_autocorr(cluster_labels,
                       X, stages, stat='pearsonr'):
    all_autocorr_coefs = []
    cluster_ids = []
    dates_1_all = []
    dates_2_all = []
    for cluster_id in np.unique(cluster_labels):
        avg_wfs = average_waveforms(cluster_id, cluster_labels,
                                    X, stages)
        autocorr_coefs, p_values, dates1, dates2 = autocorrelation(avg_wfs, stat=stat)
        dates_1_all.extend(dates1)
        dates_2_all.extend(dates2)
        cluster_ids.extend([cluster_id for _ in range(len(autocorr_coefs))])
        all_autocorr_coefs.extend(autocorr_coefs)
    return all_autocorr_coefs, cluster_ids, dates_1_all, dates_2_all

def all_cross_corr(cluster_labels,
                    X, stages, combs, stat='pearsonr'):
    all_cross_corr = []
    cluster_ids = []
    all_dates = []
    for cluster_id1, cluster_id2 in combinations(np.unique(cluster_labels), r=2):
        if (cluster_id1, cluster_id2) in combs:
            avg_wfs1 = average_waveforms(cluster_id1, cluster_labels,
                                         X, stages)
            avg_wfs2 = average_waveforms(cluster_id2, cluster_labels,
                                         X, stages)
            corr_coefs, p_values, dates = cross_correlations(avg_wfs1, avg_wfs2, stat=stat)
            cluster_ids.extend([(cluster_id1, cluster_id2) for _ in range(len(corr_coefs))])
            all_cross_corr.extend(corr_coefs)
            all_dates.extend(dates)
    return all_cross_corr, cluster_ids, all_dates
        
def mix_colors(c1, c2):
    c1, c2 = Color(hex=c1), Color(hex=c2)
    c_new = tuple((c1.rgb[i]+c2.rgb[i])/2 for i in range(3))
    c_new = Color(c_new)
    return c_new.hex
    
def cross_correlation_hist(cluster_labels,
                           X, stages, color_dic, ax,
                           min_corr=0.2, nb_bins=400, combs = [], stat='pearsonr'):
    bins = np.linspace(min_corr, 1., nb_bins)
    p = 0
    for cluster_id1, cluster_id2 in combinations(np.unique(cluster_labels), r=2):
        if  (cluster_id1, cluster_id2) in combs:
            avg_wfs1 = average_waveforms(cluster_id1, cluster_labels,
                                         X, stages)
            avg_wfs2 = average_waveforms(cluster_id2, cluster_labels,
                                         X, stages)
            corr_coefs, p_values, dates = cross_correlations(avg_wfs1, avg_wfs2)
            h, b = np.histogram(corr_coefs, bins=bins, density=False)
            h = (h/len(corr_coefs))*100
            #c = mix_colors(color_dic[int(cluster_id1)], color_dic[int(cluster_id2)])
            if p == 0:
                ax.step(bins[:-1], h, color='black', alpha=0.5, lw=0.5, label='cross correlation')
                p += 1
            else:
                ax.step(bins[:-1], h, color='black', alpha=0.5, lw=0.5)
    
    if stat == 'pearsonr':
        ax.set_xlabel('Pearson correlation coefficient')
    elif stat == 'spearmanr':
        ax.set_xlabel('Spearman correlation coefficient')
    ax.set_ylabel('Percentage of total comparisons %')
    return ax   

def correlation_plot(df,
                     mouse,
                     neuron_key,
                     neuron_color_dic,
                     spike_cols,
                     combs,
                     savefig,
                     savefigpath,
                     min_corr=0.3,
                     min_corr_zoom=0.8,
                     stat='pearsonr',
                     date_key='date',
                     zoom=False):
    #select df
    df = df[df['mouse'] == mouse]
    color_dic = dict([(int(key), val) for key, val in neuron_color_dic.items()])
    #create figure
    fig = plt.figure(figsize=(10, 5))
    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)
    f2_ax1 = fig.add_subplot(spec[:, :2])
    f2_ax2 = fig.add_subplot(spec[:, 2])
    #plot main histogram figure
    f2_ax1 = autocorr_histogram(df[neuron_key].values,
                            df[spike_cols].values,
                            df[date_key].values,
                            color_dic,
                            f2_ax1, min_corr=min_corr, nb_bins=300, stat=stat)
    if zoom:
        f2_ax1.set_xlim(min_corr_zoom, 1)
    #get data for swarmplots, i.e all auto and cross correlation pearson coefficients
    all_autocorr_coefs, neuron_ids_autocorr, dates1, dates2 = all_autocorr(df[neuron_key].values,
                        df[spike_cols].values,
                        df[date_key].values,
                        stat=stat)
    all_crosscorr_coefs, neuron_ids_crosscorr, cross_dates = all_cross_corr(df[neuron_key].values,
                        df[spike_cols].values,
                        df[date_key].values, combs,
                        stat=stat)
    #define dataframes to work with seaborn
    df_plot = pd.DataFrame(all_autocorr_coefs, columns=['autocorr_coef'])
    df_plot['type'] = ['autocorr' for _ in range(len(all_autocorr_coefs))]
    df_plot['neuron'] = neuron_ids_autocorr
    df_plot['date_1'] = dates1
    df_plot['date_2'] = dates2
    df_plot2 = pd.DataFrame(all_crosscorr_coefs, columns=['cross_coef'])
    df_plot2['type_'] = ['crosscorr' for _ in range(len(all_crosscorr_coefs))]
    df_plot2['neuron'] = neuron_ids_crosscorr
    df_plot2[date_key] = cross_dates
    #plot box plots
    min_y = min(np.min(df_plot["autocorr_coef"]), np.min(df_plot2["cross_coef"])) - 0.05
    f2_ax2 = sns.swarmplot(x="type", y="autocorr_coef", data=df_plot, s=0.2, ax=f2_ax2)
    f2_ax2 = sns.boxplot(x="type", y="autocorr_coef", data=df_plot, color="indianred", ax=f2_ax2)
    f2_ax2.set_ylabel("")
    f2_ax2.set_title("Auto-correlation")
    f2_ax2.set_xticks([])
    f2_ax2.set_xlabel("")
    f2_ax2.set_ylim((min_y, 1))
    
    f2_ax1.legend(loc="upper left")
    fig.tight_layout()
    #savefig
    savefig = True
    if savefig:
        plt.savefig(savefigpath + '.pdf', format='pdf')
        plt.savefig(savefigpath + '.jpg', format='jpg')
    plt.show()
    return df_plot, df_plot2


def autocorrelation_plot(df_plot, x_key, binwidth=0.01,
                         stat='probability', savefig_path=None):
    '''
    Simple autocorrelation histogram based on output of correlation_plot function
    Args:
        df_plot (pandas DataFrame)
        x_key (str): column identifier in df_plot for x axis in histogram
        binwidth (float)
        stat (str): passed to sns.histplot
        savefig_path (str): passed to plt.savefig
    '''
    ax = sns.histplot(x=x_key, data=df_plot, binwidth=binwidth, stat=stat)
    if savefig_path:
        plt.savefig(savefig_path)
    return ax

def correlations(df,
                mouse,
                neuron_key,
                spike_cols,
                combs,
                stat='pearsonr',
                date_key='date'):
    df = df[df['mouse'] == mouse]
    all_autocorr_coefs, neuron_ids_autocorr, dates1, dates2 = all_autocorr(df[neuron_key].values,
                        df[spike_cols].values,
                        df[date_key].values,
                        stat=stat)
    all_crosscorr_coefs, neuron_ids_crosscorr, cross_dates = all_cross_corr(df[neuron_key].values,
                        df[spike_cols].values,
                        df[date_key].values, combs,
                        stat=stat)
     #define dataframes to work with seaborn
    df_plot = pd.DataFrame(all_autocorr_coefs, columns=['autocorr_coef'])
    df_plot['type'] = ['autocorr' for _ in range(len(all_autocorr_coefs))]
    df_plot['neuron'] = neuron_ids_autocorr
    df_plot['date_1'] = dates1
    df_plot['date_2'] = dates2
    df_plot2 = pd.DataFrame(all_crosscorr_coefs, columns=['cross_coef'])
    df_plot2['type_'] = ['crosscorr' for _ in range(len(all_crosscorr_coefs))]
    df_plot2['neuron'] = neuron_ids_crosscorr
    df_plot2[date_key] = cross_dates
    return df_plot, df_plot2