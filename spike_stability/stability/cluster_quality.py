from numpy.core.arrayprint import DatetimeFormat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score, r2_score
from spike_stability.util.quality_metrics import give_me_metrics,  plot_metrics
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression


def calc_sil_score(df,
                   mouse,
                   channel,
                   date_key=date_key,
                   channel_key=channel_key,
                   coord_keys=['UMAP 1', 'UMAP 2'],
                   cluster_key='clusters Wavemap day'):
    '''
    Calculate silhouette score of clustering in low dimensional space
    for a certain channel
    Args:
        df (pandas dataframe)
        mouse (int)
        channel (int)
        coord_keys (list of str)
        cluster_key (str)
    Returns:
        scores (list): list of silhouette scores for each timepoint
        dates (list)
    '''
    df_ = df[(df['mouse'] == mouse) & (df[channel_key] == channel)]
    scores = []
    dates = []
    for date in df[date_key].unique():
        df_1 = df_[df_[date_key] == date]
        X_1 = df_1[coord_keys].values
        y_1 = df_1[cluster_key].values
        if len(X_1)>=2:
            if len(np.unique(y_1)) > 1:
                dates.append(date)
                sil_scores_1 = silhouette_score(X_1, y_1)
                scores.append(sil_scores_1)
    return scores, dates

def l_ratio_stats_2(data, all_dates):
    for key, val in data.items():
        if 'dates' in key:
            missing_index = np.array(all_dates)[~np.in1d(all_dates, val)]
            print(missing_index)
            for el in missing_index:
                data['L-ratio_channel'+str(key.split('channel')[1])].insert(el, np.NaN)
                data['sil_score_channel'+str(key.split('channel')[1])].insert(el, np.NaN)
    all_l_ratio = [val for key, val in data.items() if 'L-ratio' in key]
    all_sil_score = [val for key, val in data.items() if 'sil_score' in key]
    means_l_ratio = np.nanmean(np.array(all_l_ratio), axis=0)
    means_sil_score = np.nanmean(np.array(all_sil_score), axis=0)
    stdev_l_ratio = np.nanstd(np.array(all_l_ratio), axis=0)
    stdev_sil_score = np.nanstd(np.array(all_sil_score), axis=0)
    return means_l_ratio, stdev_l_ratio, means_sil_score, stdev_sil_score


def insert_NaN_values(data, all_dates):
    for key, val in data.items():
        if 'dates' in key:
            missing_index = np.array(all_dates)[~np.in1d(all_dates, val)]
            print(missing_index)
            for el in missing_index:
                data['L-ratio_channel'+str(key.split('channel')[1])].insert(el, np.NaN)
                data['sil_score_channel'+str(key.split('channel')[1])].insert(el, np.NaN)
    return data

#this function is similarly not generalizable              
def l_ratio_sil_score(df,
                              mouse,
                              channel_key='',
                              date_key='',
                              cluster_key='clusters Waveclus',
                              coord_keys=['PCA 1', 'PCA 2']):
    all_l_ratio = []
    all_sil_scores = []
    df_ = df[df['mouse'] == mouse]
    unique_dates = sorted(list(df_[date_key].unique()))
    unique_dates_ = list(np.arange(len(df_[date_key].unique()))+1)
    data = {}
    for i, channel in enumerate(df_[channel_key].unique()):
        #select subset of df
        print(channel)
        df_c = df_[df_[channel_key] == channel]
        X = df_c[coord_keys].values
        cluster_labels = df_c[cluster_key].values
        #only calculate quality metric if more than one cluster is present for this channel
        if len(np.unique(cluster_labels)) > 1:
            dates = df_c[date_key].values
            dates_mapping = dict([(val, unique_dates.index(val)+1) for i, val in enumerate(sorted(np.unique(dates)))])
            print(dates_mapping)
            dates_int = list(map(dates_mapping.get, dates))
            sorted_dates = sorted(np.unique(dates_int), key=int)
            #get metrics for both
            #l_ratio
            l_ratios, isolation_distances, nn_hit_rate, nn_miss_rate, cluster_labels = give_me_metrics(X,
                                                                                                      cluster_labels,
                                                                                                      dates_int)
            metrics = [l_ratios, isolation_distances, nn_hit_rate, nn_miss_rate]
            averaged_l_ratio = [np.nanmean(l_ratios[date]) for date in np.unique(dates_int)]
            #silhouette score
            sil_scores, dates_sil = calc_sil_score(df, mouse, channel, coord_keys=coord_keys,
                                                    date_key=date_key,
                                                    channel_key=channel_key,cluster_key=cluster_key)
            all_l_ratio.append(averaged_l_ratio)
            all_sil_scores.append(sil_scores)
            data['L-ratio_channel'+str(channel)] = averaged_l_ratio
            data['sil_score_channel'+str(channel)] = sil_scores
            data['dates_channel'+str(channel)] = sorted_dates

    data = insert_NaN_values(data, unique_dates_)
    data_df = format_data_to_df(data)
    return data_df 


def format_data_to_df(data):
    L_ratios = np.hstack([val for key, val in data.items() if 'L-ratio' in key])
    sil_scores = np.hstack([val for key, val in data.items() if 'sil_score' in key])
    dates = np.hstack([val for key, val in data.items() if 'dates' in key])
    channels = np.hstack([[int(key.split('channel')[1]) for _ in val] for key, val in data.items() if 'dates' in key])
    data_df = pd.DataFrame(L_ratios, columns=['L-ratio'])
    data_df['dates'] = dates
    data_df['channels'] = channels
    data_df['sil_score'] = sil_scores
    return data_df


def plot_quality_metrics_from_df(data_df):
    sns.lineplot(x='dates', y='L-ratio', data=data_df, palette='flare',
             hue='channels', alpha=0.2, marker='o')
    sns.lineplot(x='dates', y='L-ratio', data=data_df, palette='flare', marker='o')
    plt.show()
    sns.lineplot(x='dates', y='sil_score', data=data_df, palette='flare',
             hue='channels', alpha=0.2, marker='o')

    sns.lineplot(x='dates', y='sil_score', data=data_df, palette='flare', marker='o')
    plt.show()