import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import chi2 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def mahalanobis_metrics(all_pcs, all_labels, this_unit_id):

    """ Calculates isolation distance and L-ratio (metrics computed from Mahalanobis distance)
    Based on metrics described in Schmitzer-Torbert et al. (2005) Neurosci 131: 1-11
    Inputs:
    -------
    all_pcs : numpy.ndarray (num_spikes x PCs)
        2D array of PCs for all spikes
    all_labels : numpy.ndarray (num_spikes x 0)
        1D array of cluster labels for all spikes
    this_unit_id : Int
        number corresponding to unit for which these metrics will be calculated
    Outputs:
    --------
    isolation_distance : float
        Isolation distance of this unit
    l_ratio : float
        L-ratio for this unit
    """

    pcs_for_this_unit = all_pcs[all_labels == this_unit_id,:]
    pcs_for_other_units = all_pcs[all_labels != this_unit_id, :]

    mean_value = np.expand_dims(np.mean(pcs_for_this_unit,0),0)

    try:
        VI = np.linalg.inv(np.cov(pcs_for_this_unit.T))
    except np.linalg.linalg.LinAlgError: # case of singular matrix
        return np.nan, np.nan

    mahalanobis_other = np.sort(cdist(mean_value,
                       pcs_for_other_units,
                       'mahalanobis', VI = VI)[0])

    mahalanobis_self = np.sort(cdist(mean_value,
                             pcs_for_this_unit,
                             'mahalanobis', VI = VI)[0])

    n = np.min([pcs_for_this_unit.shape[0], pcs_for_other_units.shape[0]]) # number of spikes

    if n >= 2:

        dof = pcs_for_this_unit.shape[1] # number of features

        l_ratio = np.sum(1 - chi2.cdf(pow(mahalanobis_other,2), dof)) / \
                mahalanobis_self.shape[0] # normalize by size of cluster, not number of other spikes
        isolation_distance = pow(mahalanobis_other[n-1],2)

    else:
        l_ratio = np.nan
        isolation_distance = np.nan

    return isolation_distance, l_ratio 

def lda_metrics(all_pcs, all_labels, this_unit_id):

    """ Calculates d-prime based on Linear Discriminant Analysis
    Based on metric described in Hill et al. (2011) J Neurosci 31: 8699-8705
    Inputs:
    -------
    all_pcs : numpy.ndarray (num_spikes x PCs)
        2D array of PCs for all spikes
    all_labels : numpy.ndarray (num_spikes x 0)
        1D array of cluster labels for all spikes
    this_unit_id : Int
        number corresponding to unit for which these metrics will be calculated
    Outputs:
    --------
    d_prime : float
        Isolation distance of this unit
    l_ratio : float
        L-ratio for this unit
    """

    X = all_pcs

    y = np.zeros((X.shape[0],),dtype='bool')
    y[all_labels == this_unit_id] = True

    lda = LDA(n_components=1)

    X_flda = lda.fit_transform(X, y)

    flda_this_cluster  = X_flda[np.where(y)[0]]
    flda_other_cluster = X_flda[np.where(np.invert(y))[0]]

    d_prime = (np.mean(flda_this_cluster) - np.mean(flda_other_cluster))/np.sqrt(0.5*(np.std(flda_this_cluster)**2+np.std(flda_other_cluster)**2))

    return d_prime

def nearest_neighbors_metrics(all_pcs, all_labels, this_unit_id, max_spikes_for_nn, n_neighbors):

    """ Calculates unit contamination based on NearestNeighbors search in PCA space
    Based on metrics described in Chung, Magland et al. (2017) Neuron 95: 1381-1394
    Inputs:
    -------
    all_pcs : numpy.ndarray (num_spikes x PCs)
        2D array of PCs for all spikes
    all_labels : numpy.ndarray (num_spikes x 0)
        1D array of cluster labels for all spikes
    this_unit_id : Int
        number corresponding to unit for which these metrics will be calculated
    max_spikes_for_nn : Int
        number of spikes to use (calculation can be very slow when this number is >20000)
    n_neighbors : Int
        number of neighbors to use
    Outputs:
    --------
    hit_rate : float
        Fraction of neighbors for target cluster that are also in target cluster
    miss_rate : float
        Fraction of neighbors outside target cluster that are in target cluster
    """

    total_spikes = all_pcs.shape[0]
    ratio = max_spikes_for_nn / total_spikes
    this_unit = all_labels == this_unit_id

    X = np.concatenate((all_pcs[this_unit,:], all_pcs[np.invert(this_unit),:]),0)

    n = np.sum(this_unit)

    if ratio < 1:
        inds = np.arange(0,X.shape[0]-1,1/ratio).astype('int')
        X = X[inds,:]
        n = int(n * ratio)


    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    this_cluster_inds = np.arange(n)

    this_cluster_nearest = indices[:n,1:].flatten()
    other_cluster_nearest = indices[n:,1:].flatten()

    hit_rate = np.mean(this_cluster_nearest < n)
    miss_rate = np.mean(other_cluster_nearest < n)

    return hit_rate, miss_rate

def give_me_metrics(X,
                    cluster_labels,
                    dates,
                    randomized=False):
    l_ratios = {}
    isolation_distances = {}
    nn_hit_rate = {}
    nn_miss_rate = {}
    max_spikes_for_nn = 100
    n_neighbors = 5

    if randomized:
        cluster_labels = np.random.permutation(cluster_labels)
    
    for date in sorted(np.unique(dates), key=int):        
        mask = np.array(dates) == date
        if np.sum(mask) >= 5:
            X_current = X[mask]
            labels_current = cluster_labels[mask]

            l_ratios_per_cluster = []
            isolation_distances_per_cluster = []
            nn_hit_rates_per_cluster = []
            nn_miss_rates_per_cluster = []
            for cluster_label in np.unique(cluster_labels):
                # we will calculate l-ratio in original waveform space but labelled with umap projection - no need for pc here
                #we could do it with pca but since clusters weren't determined with pca representation it'll give terrible results
                isolation_distance_, l_ratio_ = mahalanobis_metrics(X_current, labels_current, cluster_label)
                nn_hit_rate_, nn_miss_rate_ = nearest_neighbors_metrics(X_current,
                                                                        labels_current, 
                                                                        cluster_label,
                                                                        max_spikes_for_nn,
                                                                        n_neighbors)

                l_ratios_per_cluster.append(l_ratio_)
                isolation_distances_per_cluster.append(isolation_distance_)
                nn_hit_rates_per_cluster.append(nn_hit_rate_)
                nn_miss_rates_per_cluster.append(nn_miss_rate_)

            l_ratios[date] = l_ratios_per_cluster
            isolation_distances[date] = isolation_distances_per_cluster
            nn_hit_rate[date] = nn_hit_rates_per_cluster
            nn_miss_rate[date] =  nn_miss_rates_per_cluster
        
    return l_ratios, isolation_distances, nn_hit_rate, nn_miss_rate, cluster_labels

#look at metrics overall and not per timepoint
def give_me_overall_metrics(X,
                            clusters,
                            randomized=False):
    
    l_ratios_per_cluster = []
    isolation_distances_per_cluster = []
    nn_hit_rates_per_cluster = []
    nn_miss_rates_per_cluster = []
    d_prime_per_cluster = []
    max_spikes_for_nn = 100
    n_neighbors = 5
    
    for cluster_label in np.unique(clusters):
        #for umap representation
        if randomized:
            cluster_labels = np.random.permutation(clusters)
        else:
            cluster_labels = clusters
            
        isolation_distance_, l_ratio_ = mahalanobis_metrics(X, cluster_labels, cluster_label)

        #we will caculate this metric in umap space
        nn_hit_rate_, nn_miss_rate_ = nearest_neighbors_metrics(X,
                                                                cluster_labels, 
                                                                cluster_label,
                                                                max_spikes_for_nn,
                                                                n_neighbors)


        l_ratios_per_cluster.append(l_ratio_)
        isolation_distances_per_cluster.append(isolation_distance_)
        nn_hit_rates_per_cluster.append(nn_hit_rate_)
        nn_miss_rates_per_cluster.append(nn_miss_rate_)
        
    metrics = [l_ratios_per_cluster, isolation_distances_per_cluster, nn_hit_rates_per_cluster, nn_miss_rates_per_cluster]
    return metrics

def plot_overall_metrics(X,
                         clusters,
                         label='WaveMAP',
                         compare_randomized=True):
    metrics = give_me_overall_metrics(X, clusters, randomized=False)
    if compare_randomized:
        metrics_random = give_me_overall_metrics(X, clusters, randomized=True)
    names = ['L_ratio', 'Isolation distance', 'nn_hit_rate', 'nn_miss_rate']
    for i, metric in enumerate(metrics):
        plt.scatter(np.unique(clusters), metric, label=label+' clusters')
        if compare_randomized:
            plt.scatter(np.unique(clusters), metrics_random[i], label='Random clusters')
        plt.xlabel('Cluster #')
        plt.ylabel(names[i])
        plt.title(f'{names[i]} per cluster over all recordings')
        if compare_randomized:
            plt.legend()
        plt.show()  


def plot_metrics(X,
                 cluster_labels,
                 dates,
                 randomized=False,
                 color_dic={},
                 ylims=[(0, 1), (0, 100), (0, 1), (0, 1)]):
    #can't be bothered to automate this and make it cleaner so let's keep it simple and messy
    l_ratios_, isolation_distances_, nn_hit_rate_, nn_miss_rate_, cluster_labels_ = give_me_metrics(X,
                                                                                                    cluster_labels,                                                                                                            
                                                                                                    dates,                                                                                              
                                                                                                    randomized=randomized)
    metrics = [l_ratios_, isolation_distances_, nn_hit_rate_, nn_miss_rate_]
    names = ['L_ratio', 'Isolation distance', 'nn_hit_rate', 'nn_miss_rate']
    for i in range(len(metrics)):
        fig = plt.figure(figsize=(20, 10))
        for el in np.unique(cluster_labels_):
            if '0' in np.unique(cluster_labels_).astype('str'):
                metric_across_days_per_cluster = [metrics[i][date][int(el)] for date in sorted(l_ratios_.keys(), key=int)]
            else:
                metric_across_days_per_cluster = [metrics[i][date][int(el)-1] for date in sorted(l_ratios_.keys(), key=int)]
            plt.scatter(sorted(l_ratios_.keys(), key=int),  metric_across_days_per_cluster, label='cluster n°'+str(el), c=color_dic[int(el)])

        plt.xlabel('Timepoint')
        plt.ylim(ylims[i][0], ylims[i][1])
        plt.ylabel(names[i])
        if randomized:
            plt.title(f'{names[i]} per cluster for each timepoint, randomized')
        else:
            plt.title(f'{names[i]} per cluster for each timepoint')
        #plt.legend() 
        plt.show() 