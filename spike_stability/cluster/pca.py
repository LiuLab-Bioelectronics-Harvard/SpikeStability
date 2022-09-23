import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def pca_day(df, spike_cols,
            prior_cluster_key,
            date_key,
            new_columns_coord_key,
            cluster_colors, plot=False, figsize=(10, 4)):
    '''
    Function to compute day by day pca representation based off spikes.
    Args:
        df (pandas DataFrame)
        spike_cols (list of str)
        prior_cluster_key (str)
        date_key (str)
        new_columns_coord_key (str)
        cluster_colors (dict)
        figsize (tuple)
    Returns:
        all_pca_coordinates (2D array)
    '''
    if plot:
        fig, axs = plt.subplots(2, len(df[date_key].unique())//2, figsize=figsize)
    df[new_columns_coord_key] = 0
    for i, day in enumerate(df[date_key].unique()):
        mask = df[date_key] == day
        spikes_ = df.loc[mask, spike_cols].values
        prior_clusters = df.loc[mask, prior_cluster_key].values
        pca_coordinates = PCA(n_components=2).fit_transform(spikes_)
        df.loc[mask, new_columns_coord_key] = pca_coordinates
        if plot:
            axs[i%2, i//2].scatter(pca_coordinates[:, 0],
                                pca_coordinates[:, 1],
                                c=list(map(cluster_colors.get, prior_clusters)))
            axs[i%2, i//2].title('Day '+str(day))
            axs[i%2, i//2].scatter(pca_coordinates[:, 0],
                                pca_coordinates[:, 1])

    return df
