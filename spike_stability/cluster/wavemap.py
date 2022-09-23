import numpy as np
import leidenalg as la
import igraph as ig
import umap
import matplotlib.pyplot as plt
import pandas as pd
import logging as logg

from sklearn.metrics import mean_squared_error
from natsort import natsorted
from itertools import permutations


def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except KeyError:
        pass
    if g.vcount() != adjacency.shape[0]:
        logg.warning(
            f'The constructed graph has only {g.vcount()} nodes. '
            'Your adjacency matrix contained redundant nodes.'
        )
    return g


def clusters_from_igraph(g_, res, seed=0):
    '''
    Given igraph return cluster assignment given by leiden clustering
    as well as associated modularity score.
    Args:
        g_ (igraph)
        res (float): resolution for leiden clustering algorithm
    Returns:
        clusters (list)
        q (float)
    '''
    partition = la.find_partition(g_,
                                  la.RBConfigurationVertexPartition,
                                  seed=seed,
                                  resolution_parameter=res)
    groups = np.array(partition.membership)
    clusters =  pd.Categorical(
            values=groups.astype('U'),
            categories=natsorted(map(str, np.unique(groups))))
    
    #calculate modularity with 'normal' modularity score
    modularity_part = la.ModularityVertexPartition(g_,
                                                   initial_membership=partition.membership)
    q = modularity_part.quality()
    return clusters, q


def pick_res(graph_, res_list, nb_clust=3, verbose=0):
    '''
    Args:
        graph_ (igraph)
        res_list (list of floats): resolutions to try for leiden algorithm
        nb_clust (int): prior known number of clusters to obtain
    Returns:
        res (float): final resolution that obtains nb_clust 
        cluster_ (list): cluster labels
        mod_ (float): associated modularity score

    '''
    if verbose:
        def verboseprint(string, *args, **kwargs):
            print(string, *args, **kwargs)
    else:
        verboseprint = lambda *a, **k: None
    for res in res_list:
        verboseprint('calculating '+str(res))
        g_ = get_igraph_from_adjacency(graph_)
        clusters_, mod_ = clusters_from_igraph(g_, res, seed=0)
        if len(np.unique(clusters_)) == nb_clust:
            return res, clusters_, mod_
        elif len(np.unique(clusters_)) > nb_clust:
            verboseprint('Not able to find resolution giving three clusters, returning starting again with smaller resolutions')
            for res_2 in np.arange(0.01, res, 0.01):
                g_ = get_igraph_from_adjacency(graph_)
                clusters_, mod_ = clusters_from_igraph(g_, res_2, seed=0)
                if len(np.unique(clusters_)) == nb_clust:
                    return res, clusters_, mod_
                elif len(np.unique(clusters_)) > nb_clust:
                    verboseprint('Not adequate number of clusters identifiable, returning current partition')
                    return res, clusters_, mod_

def auto_match(templates, candidates):
    '''
    Match candidate average waveforms with templates.
    Args:
        templates(2D array or list of arrays)
        candidates (2D array or list of arrays)
    Returns:
        optimal_combination (list of tuples): contains
        matchings e.g [(1, 2), (2, 3) ...] this means 1st template
        matches 2nd candidate and so on
    '''
    min_mse = 1e10
    optimal_combination = []
    unique_combinations = [list(zip(range(1, len(templates)+1), p)) for p in permutations(range(1, len(candidates)+1))]
    for combination in unique_combinations:
        mse_ = 0
        for pair in combination:
            mse_ += mean_squared_error(templates[pair[0]-1], candidates[pair[1]-1])
        if mse_ <= min_mse:
            min_mse = mse_
            optimal_combination = combination    
    return optimal_combination

def align_umap_day_by_day(spikes, dates, clusters_umap_per_day, 
                          cluster_colors_umap_day, day_align, plot=True, verbose=0):
    '''
    Align cluster labels across days by minimizing mean squared error between pairings.
    Args:
        spikes (2D array)
        dates (1D array of ints)
        clusters_umap_per_day (1D array)
        cluster_colors_umap_day (color dict)
        umap_params (dict): parameters to pass to umap function
        align_day (int or same type as dates): day for template for alignment of wavemap day by day produced clusters
        verbose (int): verbosity level
    Returns:
        correct_clusters_umap_per_day (1D array): aligned clusters
    '''
    if verbose:
        def verboseprint(string, *args, **kwargs):
            print(string, *args, **kwargs)
    else:
        verboseprint = lambda *a, **k: None
    correct_clusters_umap_per_day = {}
    #get template waveforms for automatic alignment
    spikes_day_align = spikes[dates == day_align]
    clusters_day_align = clusters_umap_per_day[day_align]
    templates = np.array([np.mean(spikes_day_align[clusters_day_align == clst], axis=0) for clst in sorted(np.unique(clusters_day_align), key=int)])
    verboseprint('Aligning wavemap labels for consistent labelling across days!')
    for date in np.unique(dates):
        mask_date = dates == date
        spikes_ = spikes[mask_date]
        #assignment
        candidates = []
        for cluster in np.unique(clusters_umap_per_day[date]):
            mask_cluster = np.array(clusters_umap_per_day[date]) == cluster
            spikes_cluster = spikes_[mask_cluster]
            candidates.append(np.mean(spikes_cluster, axis=0))
        optimal_match = auto_match(templates, candidates)
        #build mapping dic
        mapping_dic = {}
        for pair in optimal_match:
            mapping_dic[str(pair[1]-1)] = int(pair[0]-1)
        verboseprint(f'Optimal match found for day {date}: {mapping_dic}')
        if plot:
            #plotting and cluster id with optimal assignment
            fig, axs = plt.subplots(1, len(np.unique(clusters_umap_per_day[date])), figsize=(30, 10))
            fig.suptitle(f'Average waveform per cluster on timepoint {date}')
            for j, cluster in enumerate(np.unique(clusters_umap_per_day[date])):
                mask_cluster = np.array(clusters_umap_per_day[date]) == cluster
                spikes_cluster = spikes_[mask_cluster]
                t = np.arange(0, spikes_cluster.shape[1])
                if len(np.unique(clusters_umap_per_day[date])) > 1:
                    axs[j].plot(t,
                                np.mean(spikes_cluster, axis=0),
                                color=cluster_colors_umap_day[mapping_dic[str(cluster)]])
                else:
                    axs.plot(t,
                            np.mean(spikes_cluster, axis=0),
                            color=cluster_colors_umap_day[mapping_dic[str(cluster)]])
            plt.show()
        correct_clusters_umap_per_day[date] = list(map(mapping_dic.get, clusters_umap_per_day[date]))
    return correct_clusters_umap_per_day

def wavemap_channel(spikes,
                    dates,
                    cluster_colors={},
                    res_list=np.arange(0.1, 5.0, 0.1),
                    nb_clust=2,
                    umap_params={'n_neighbors': 20,
                                    'random_state': 1,
                                    'min_dist': 0.1, 
                                    'n_components': 2,
                                    'metric': 'euclidean'},
                    align_day=1,
                    verbose=0,
                    automatic_cluster_nb=False,
                    plot=True):
    '''
    Compute day by day umap whilst performing leiden algorithm and resolution tuning
    to select the number of clusters chosen by the user.
    Args:
        spikes (2D array)
        dates (1D array of ints)
        cluster_colors (dict): mapping from cluster label to hex color
        res_list (list or 1D array of floats)
        nb_clust (int)
        umap_params (dict): parameters to pass to umap function
        align_day (int or same type as dates): day for template for alignment of wavemap day by day produced clusters
        verbose (int): verbosity level
        plot (bool): include plots for clustering and dimension reduction produced each day
    Returns:
        X_umap_per_day (2D array)
        clusters_umap_per_day (1D array): aligned clusters
        res_per_day (list)
        mod_per_day (list)
    '''
    X_umap_per_day = {}
    clusters_umap_per_day = {}
    res_per_day = {}
    mod_per_day = {}
    if str(np.unique(dates)[0]).isdigit(): #whether str or int, will give correct order for dates if 1, 2, 3, ..
        unique_dates = sorted(np.unique(dates), key=lambda x: int(x))
    else: #can change this to allow for %m%d%y ordering of dates
        unique_dates = np.unique(dates)
    for date in unique_dates:
        mask_date = dates == date
        spikes_ = spikes[mask_date]
        #get umap representation
        mapper_ = umap.UMAP(n_neighbors=umap_params['n_neighbors'],
                            random_state=umap_params['random_state'], 
                            min_dist=umap_params['min_dist'],
                            n_components=umap_params['n_components'],
                            metric=umap_params['metric']).fit(spikes_)
        X_umap_per_day[date] = mapper_.transform(spikes_)

        #graph and leiden clustering
        try:
            if automatic_cluster_nb:
                g_ = get_igraph_from_adjacency(mapper_.graph_)
                clusters_, mod_ = clusters_from_igraph(g_, 0.1, seed=0)
                res = 0.1
            else:
                res, clusters_, mod_ = pick_res(mapper_.graph_, res_list, nb_clust=nb_clust, verbose=verbose)
            res_per_day[date] = res
            clusters_umap_per_day[date] = clusters_
            mod_per_day[date] = mod_
            if plot:
                #plot
                plt.scatter(X_umap_per_day[date][:, 0],
                            X_umap_per_day[date][:, 1],
                            s=0.5,
                            c=list(map(cluster_colors.get, clusters_.astype('int'))))

                plt.xticks([])
                plt.yticks([])
                plt.xlabel('UMAP 1', fontsize=10)
                plt.ylabel('UMAP 2', fontsize=10)
                plt.title(f'Timepoint {date} umap representation', fontsize=12)
                plt.show()
        except Exception as e:
            print(f'Finding leiden resolution failed for date {date} due to {e}')
    
    #now align the cluster labels

    clusters_umap_per_day = align_umap_day_by_day(spikes, dates, clusters_umap_per_day, cluster_colors, align_day, plot=plot, verbose=verbose)
    return X_umap_per_day, clusters_umap_per_day, res_per_day, mod_per_day

def wavemap(df,
            spike_cols=[],
            date_key='dates',
            channel_key='channel',
            new_coordinates_key=['UMAP1_day', 'UMAP2_day'],
            new_cluster_labels_key='cluster_wavemap',
            cluster_colors_channel='',
            nb_clust_channel={},
            align_day_channel={},
            verbose=0,
            **kwargs):
    '''
    Perform leiden clustering on umap representations per channel. This supposes being sure
    no neuron is being recorded on two separate channels, this is not the case for ultra dense 
    electrode arrays.
    This also supposes that data is correctly ordered by channel and date. This means that in
    the arrays, all values corresponding to the same channel are next to each other and within a
    channel the order is by incresing date.
    Args:
        df (pandas DataFrame)
        spike_cols (list of str)
        date_key (str): column identifier of dates in df
        channel_key (str): column identifier of channel in df
        new_coordinates_key (list of str): coordinate column identifiers to add to df with umap embeddings
        new_cluster_labels_key (str): column identifier of cluster label
        cluster_colors_channel (dict of dict): mapping dict for cluster label to color per channel
        nb_clust_channel (dict)
        align_day_channel (dict)
        **kwargs : arguments passed to wavemap_channel
    Returns:
        df (pandas DataFrame)
    '''
    if verbose:
        def verboseprint(string, *args, **kwargs):
            print(string, *args, **kwargs)
    else:
        verboseprint = lambda *a, **k: None
    df[new_coordinates_key] = 0
    df[new_cluster_labels_key] = 0
    verboseprint('Processing in this order: '+str(df[channel_key].unique()))
    for i, channel in enumerate(df[channel_key].unique()):
        #fetch channel specific data
        mask = df[channel_key] == channel
        #to ensure correct order
        spikes_, dates_ = df.loc[mask, spike_cols].values, df.loc[mask, date_key].values
        #compute umap + leiden clustering
        X_umap_, clusters_umap_, _, _ = wavemap_channel(spikes_,
                                                         dates_,
                                                         cluster_colors=cluster_colors_channel[channel],
                                                         nb_clust=nb_clust_channel[channel],
                                                         align_day=align_day_channel[channel],
                                                         verbose=verbose,
                                                         **kwargs)
        #append
        for date in df[mask][date_key].unique():
            mask2 = df[date_key] == date
            df.loc[mask&mask2, new_coordinates_key] = X_umap_[date]
            df.loc[mask&mask2, new_cluster_labels_key] = clusters_umap_[date]
    return df

def umap_channel_overall_timepoints(df, spike_cols, channel_key,
                                    new_coordinates_key=[], plot=False,
                                    umap_params={'n_neighbors': 50,
                                          'random_state': 1,
                                          'min_dist': 0.1, 
                                          'n_components': 2,
                                          'metric': 'euclidean'}):
    '''
    Calculate and append to dataframe overall umap coordinates per recording channel.
    Args:
        df (pandas dataframe)
        spike_cols (list of str)
        channel _key (str)
        new_coordinates_key (list of str)
        plot (bool)
        umap_params (dict): to pass to umap.UMAP() function of umap-learn module
    Returns:
        df with added columns
    '''
    df[new_coordinates_key] = 0
    for chan in df[channel_key].unique():
        mask = df[channel_key] == chan
        spikes_ = df.loc[mask, spike_cols].values
        mapper_ = umap.UMAP(n_neighbors=umap_params['n_neighbors'],
                                random_state=umap_params['random_state'], 
                                min_dist=umap_params['min_dist'],
                                n_components=umap_params['n_components'],
                                metric=umap_params['metric']).fit(spikes_)
        X_umap = mapper_.transform(spikes_)
        df.loc[mask, new_coordinates_key] = X_umap
        if plot:
            plt.scatter(X_umap[:, 0], X_umap[:, 1], s=1.)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.show()
    return df
