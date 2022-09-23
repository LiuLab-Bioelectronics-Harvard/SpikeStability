import numpy as np
import pandas as pd

def add_neuron_label(df, cluster_key, channel_key, new_neuron_key):
    '''
    Add new column with neuron label based on per-channel cluster labels.
    E.g. CH13 has cluster labels 1, 2, 3 and CH21 has labels 1, 2. New column
    will have labels 1, 2, 3, 4, 5 (unique id for a given mouse, regardless of channel).
    **Should only be used on dataset with one mouse data**
    Args:
        df (pandas DataFrame)
        cluster_key (str): cluster column identifier 
        channel_key (str): channel column identifier
        new_neuron_key (str): new column identifier to add to dataframe with neuron labels 
    Returns:
        df (pandas DataFrame)
    '''
    current_max_label = 0
    df[new_neuron_key] = 0
    offset = 1 - df[cluster_key].min()
    for channel in df[channel_key].unique():
        mask = df[channel_key] == channel
        df.loc[mask, new_neuron_key] = df[mask][cluster_key] + current_max_label
        current_max_label = df.loc[mask, new_neuron_key].max() + offset
    return df


def postprocessing_dataset(df, mouse, nb_points, distance,
                          date_key='date', coord_keys='', neuron_key=''):
    '''
    Postprocessing based off embedding representation.
    Selects spikes with an embedding < distance to the neuron cluster centroid
    on a given timepoint.
    Args:
        df (pandas DataFrame)
        mouse (int)
        nb_points (int)
        distance (float)
        date_key (str)
        coord_keys (list of str)
        neuron_key (str)
    Returns:
        df_postprocess (pandas DataFrame)
    '''
    df_ = df[df['mouse'] == mouse]
    X_postprocess = []
    for n in df_[neuron_key].unique():
        mask_n = df_[neuron_key] == n
        for day in df_[mask_n][date_key].unique():
            mask_d = df_[date_key] == day
            df_nd = df_[mask_n & mask_d]
            centroid = df_nd[coord_keys].mean(axis=0).values
            distance_matrix = np.linalg.norm(df_nd[coord_keys].values - centroid, axis=1)
            ind_points1 = distance_matrix.argsort()[:nb_points] 
            ind_points = distance_matrix[ind_points1] < distance
            X_postprocess.append(df_nd.iloc[ind_points1].iloc[ind_points].values)
    X_postprocess = np.vstack(X_postprocess)
    df_postprocess = pd.DataFrame(X_postprocess, columns=df.columns)
    return df_postprocess


def drop_cluster(df, channel, cluster_label, renumber=False, channel_key='', cluster_key=''):
    '''
    Drop cluster from dataframe
    Args:
        df (pd Df)
        channel (int)
        cluster_label (int): cluster to drop
        renumber (bool): whether to renumber cluster labels
        channel_key (str): column identifier of channel numbers in df
        cluster_key (str): column identifier for cluster labels in df
    Returns:
        df (pd Df): without cluster label for 
    '''
    mask_channel = df[channel_key] == channel
    mask_cluster_label = df[cluster_key] == cluster_label
    mask_drop = mask_channel & mask_cluster_label
    df = df.drop(index=mask_drop)
    if renumber:
        df.loc[mask_channel, cluster_key] = df.loc[mask_channel, cluster_key].apply(lambda x:
        x-1 if x > cluster_label else x)
    return df


def map_dates_to_int(df, date_key, new_date_col_key):
    '''
    Adds a column with integer ids for dates. Maps the str %m%d%y values to integers
    for each recording day.
    **Should only be used on dataset with one mouse data**
    Args:
        df (pandas DataFrame): either with only one mouse data or multiple data
        date_key (str): references columns with str dates in df
        new_date_col_key (str): new column title in df with integer dates
    Returns:
        df (pandas DataFrame): with new column
    '''
    unique_str_dates = sorted(df[date_key].unique()) #this will work if str in %Y-%m-%d format
    df[new_date_col_key] = df[date_key].apply(lambda x: unique_str_dates.index(x)+1)
    return df
