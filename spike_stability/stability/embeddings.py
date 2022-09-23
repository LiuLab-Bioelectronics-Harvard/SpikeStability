import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from matplotlib import animation
from matplotlib.ticker import MaxNLocator
from numpy.core.shape_base import atleast_1d
from statsmodels.multivariate.manova import MANOVA
from mpl_toolkits.mplot3d import Axes3D

def set_ax_style(ax, x_label, y_label, z_label,
                x_scale, y_scale, z_scale, background=False):
    ax.set_xlabel(x_label, labelpad=20)
    ax.set_ylabel(y_label, labelpad=20)
    ax.set_zlabel(z_label, labelpad=20)
    if not background:
        #set background to white
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.axis('off')
        #ax.yaxis('off')
        #ax.zaxis('off')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    #scale
    scale=np.diag([x_scale, y_scale, z_scale, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3]=1.0 
    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)
    ax.get_proj=short_proj
    return ax


def calculate_shift(means, stds):
    mean_std = np.mean(stds, axis=0)
    print('mean std' + str(mean_std))
    avg_shift_mean = np.mean(np.abs(np.diff(means, axis=0)), axis=0)
    neuron_shift = np.abs(np.diff(means, axis=0))
    print('avg shift mean' + str(avg_shift_mean))
    print('result')
    print('avg shift / mean' + str(avg_shift_mean/mean_std))
    print('norm :'+str(np.linalg.norm(avg_shift_mean/mean_std)))
    return [np.linalg.norm(neuron_shift/mean_std, axis=1)]
        

def postprocessing_step(df, nb_points, distance, coord_keys,
                        cluster_key, neuron_key, channel_key='channel', date_key='date'):
    '''
    Adding postprocessing to visualization if needed.
    Args:
        df (pandas dataframe): dataframe object containing coordinates and cluster labeling
        nb_points (int): number of points to keep for each neuron per day
        distance (float): max distance between points of neuron on given day and cluster centroid
        coord_keys (list of str): coordinate keys in df 
        cluster key (str): cluster labels reference in df
        neuron_key (str)
        channel_key (str)
        date_key (str)
    Returns:
        X_list (list of 2D arrays): per channel list of coordinates
        clusters_list (list of 1D arrays): per channel list of cluster labels
        dates_list (list of 1D arrays): per channel list of dates
        centroid_list (list of 2D arrays): per channel list of centroids for each neuron
        for each day 
    '''
    #initialize lists
    X_list, clusters_list, dates_list, centroid_list = [], [], [], []
    neurons_list = []
    #iterate through individual channels
    for chan in df[channel_key].unique():
        X, c, d, centroids, n = [], [], [], [], []
        for day in df[date_key].unique():
            mask = (df[channel_key] == chan) & (df[date_key] == day)
            X_ = df[mask][coord_keys].values
            labels_ = df[mask][cluster_key].values
            dates_ = df[mask][date_key].values
            neurons_ = df[mask][neuron_key].values
            #postprocess step for each neuron of current channel on a given day
            for label in np.unique(labels_):
                X_this_label = X_[labels_ == label]
                #centroid and distances 
                centroid = np.mean(X_this_label, axis=0)
                distance_matrix = np.linalg.norm(X_this_label - centroid, axis=1)
                #keep closest points to centroid
                ind_points1 = distance_matrix.argsort()[:nb_points] 
                ind_points = distance_matrix[ind_points1] < distance
                #append
                X.append(X_this_label[ind_points1][ind_points])
                c.append(labels_[labels_ == label][ind_points1][ind_points])
                d.append(dates_[labels_ == label][ind_points1][ind_points])
                n.append(neurons_[labels_ == label][ind_points1][ind_points])
                centroids.append(centroid)
        centroid_list.append(np.array(centroids))
        X_list.append(np.vstack(X))
        clusters_list.append(np.hstack(c))
        dates_list.append(np.hstack(d))
        neurons_list.append(np.hstack(n))
    return X_list, clusters_list, dates_list, centroid_list, neurons_list
   
def plot_centroids(X,
                   dates,
                   clusters,
                   ax,
                   displace,
                   color_dic,
                   dates_axis='z',
                   neuron_list=[],
                   date_correction=0,
                   verbose=0):
    '''
    Function to plot centroids of each neuron over time in 3D.
    Args:
        X (2D array): coordinates for one neuron across days
        dates (1D array of ints): date associated with each point in X
        clusters (1D array): cluster label associated with each point in X
        ax (matplotlib ax object): ax to add this plot to
        displace (displacement values to add to first coordinate of other channels
        to avoid overlapping 3D representations
        color_dic (dict): color mapping
        dates_axis (str): determine orientation of 3D plot
        date_correction (int): offset for dates if needed
    Returns:
        ax with added plot
        m1_list (list): list of consistency metric with determinant of covariance matrix 
        m2_list (list): list of 2nd consistency metric with
        Voinov and Nikulin coefficient of variation
    '''
    m1_list, m2_list = [], []
    neuron_shift_ = []
    neuron_list_ = []
    for cluster in np.unique(clusters):
        mask_cluster = clusters == cluster
        X_cluster = X[mask_cluster]
        current_neuron = neuron_list[mask_cluster]
        #initialize centroid dic
        centroids_cluster = []
        stds_cluster = []
        for day in np.unique(dates):
            mask_date = dates == day
            mask_total = mask_cluster & mask_date
            X_ = X[mask_total]
            #compute centroid for one cluster on one day
            centroid = np.mean(X_, axis=0)
            centroids_cluster.append(centroid)
            #compute std of cluster on day
            stdev = np.std(X_, axis=0)
            stds_cluster.append(stdev)
        centroids_cluster = np.array(centroids_cluster)
        stds_cluster = np.array(stds_cluster)
        if dates_axis == 'z':
            ax.plot(centroids_cluster[:, 0]+displace,
                    centroids_cluster[:, 1],
                    np.unique(dates)+date_correction,
                    color=color_dic[str(cluster)],
                    marker='.',
                    markersize=10,
                    markerfacecolor=color_dic[str(cluster)],
                    markeredgewidth=1,
                    markeredgecolor='black',
                    alpha=1.,
                    linewidth=3,
                    zorder=10)
            ax.zorder= 10
        elif dates_axis == 'y':
            ax.plot(centroids_cluster[:, 0]+displace,
                    np.unique(dates)+date_correction,
                    centroids_cluster[:, 1],
                    color=color_dic[str(cluster)],
                    marker='.',
                    markersize=10,
                    markerfacecolor=color_dic[str(cluster)],
                    markeredgewidth=1,
                    markeredgecolor='black',
                    alpha=1.,
                    linewidth=3,
                    zorder=10)
            ax.zorder = 10
        elif dates_axis == 'x':
            ax.plot(np.unique(dates)+date_correction,
                    centroids_cluster[:, 1],
                    centroids_cluster[:, 0]+displace,
                    color=color_dic[str(cluster)],
                    marker='.',
                    markersize=10,
                    markerfacecolor=color_dic[str(cluster)],
                    markeredgewidth=1,
                    markeredgecolor='black',
                    alpha=1.,
                    linewidth=3,
                    zorder=10)
        metric_det_cov_matrix = det_cov_matrix(centroids_cluster)
        multi_dim_coeff_variation = multi_dim_coeff_variation_VN(centroids_cluster)
        neuron_shift = calculate_shift(centroids_cluster, stds_cluster)  
        neuron_list_.append([current_neuron for _ in neuron_shift])
        neuron_shift_.append(neuron_shift)
        m1_list.append(metric_det_cov_matrix)
        m2_list.append(multi_dim_coeff_variation)
        if verbose:
            print('Computing cluster '+str(cluster) + ' color '+str(color_dic[str(cluster)]))
            print(neuron_shift)
            print('cov matrix det ' + str(metric_det_cov_matrix))
            print('multi dim CV ' + str(multi_dim_coeff_variation)) 
    return ax, m1_list, m2_list, neuron_shift_, neuron_list_

def stat_test(df,
              mouse:int,
              neuron_key,
              coord_keys):
    '''
    Perform MANOVA statistical test per neuron on the 2D representations
    Prints results.
    Args:
        df (pandas dataframe)
        mouse (int): mouse number being considered
        neuron_key (str)
        coord_keys (list of str)
    Returns:
    '''
    df_ = df[(df['mouse'] == mouse)]
    for neuron in df_[neuron_key].unique():
        print('Computing neuron '+str(neuron))
        df_n = df_[df_[neuron_key] == neuron]
        maov = MANOVA.from_formula(f'{coord_keys[0]} + {coord_keys[1]} ~ dates', data=df_n)
        print(maov.mv_test())
            
def det_cov_matrix(centroids):
    cov_matrix = np.cov(centroids.T)
    return np.sqrt(np.linalg.det(cov_matrix))

def multi_dim_coeff_variation_VN(centroids):
    cov_matrix = np.cov(centroids.T)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mean = np.mean(centroids, axis=0)
    return 1/np.sqrt((np.dot(mean.T, np.dot(inv_cov_matrix, mean))))

def all_channels_over_time(df,
                         mouse:int,
                         dates_axis,
                         color_dic_list,
                         savefig=True,
                         savefig_path='',
                         file_format='pdf',
                         cluster_key='clusters Wavemap day',
                         neuron_key='neuron Wavemap',
                         coord_keys=['UMAP 1 Overall', 'UMAP 2 Overall'],
                         labels=['UMAP 1', 'UMAP 2'],
                         displace=[0, 20, 27],
                         include_points=True,
                         include_links=True,
                         channel_key='channel',
                         date_key='date',
                         azim = 230,
                         elev = 15,
                         dist = 10,
                         zlims=(5, 18),
                         figsize=(30, 20),
                         date_correction=0,
                         animate=False,
                         nb_points=1000,
                         verbose=0,
                         distance=2,
                         lw=0.1,
                         postprocess=False,
                         dot_size=10.,
                         anim_save_path='../results/all/mouse1_3D_umap_anim_x_axis.gif',
                         xlim=(),
                         ylim=()):
    '''
    Function for 3D plotting with multiple options for animation or postprocessing.
    Args:
        df (pandas dataframe): should contain cluster_key and coord_key columns
        mouse (int): mouse identifier
        dates_axis (1d array with dates)
        color_dic_list (dict): mapping dict
        savefig (bool)
        savefig_path (str)
        file_format (str)
        cluster_key (str): identifier for cluster labels in df
        neuron_key (str): identifier for neuron labels in df
        coord_keys (list of str): identifiers for coordinates to plot in df
        labels (list of str): labels for axis of coordinates in 3D plot
        displace (list of ints): values to add to separate channels to make sure plots aren't overlapping
        include_points (bool): whether to include individual dots
        include_link (bool): whether to include links and cluster centroids
        azim, elev, dist, lw , dot_size; zlims, figsize: params for ax
        animate (bool): whther to save animated version of plot
        nb_points, distance : postprocessing params
        postprocess (bool): include postprocessing or not
        anim_save_path (str): path to save figure
    Returns:
        determinants of cov matrix & Voinov and Nikulin's multi dimensional coefficient
        of variation if include_points is true, if not empty lists
    '''
    df_ = df[df['mouse'] == mouse]
    #dates mapping
    dates_mapping = dict([(date, i) for i, date in enumerate(df_[date_key].unique())])
    #group
    if postprocess:
        X_list, clusters_list, dates_list, centroid_list, neurons_list = postprocessing_step(df_,
                                                                                             nb_points,
                                                                                             distance,
                                                                                             coord_keys,
                                                                                             cluster_key,
                                                                                            neuron_key,
                                                                                            channel_key=channel_key,
                                                                                            date_key=date_key)
    else:
        X_list = [df_[df_[channel_key] == chan][coord_keys].values for chan in df_[channel_key].unique()]
        clusters_list = [df_[df_[channel_key] == chan][cluster_key].values for chan in df_[channel_key].unique()]
        dates_list = [df_[df_[channel_key] == chan][date_key].values for chan in df_[channel_key].unique()]
        neurons_list = [df_[df_[channel_key] == chan][neuron_key].values for chan in df_[channel_key].unique()]
    dates_list = [np.array(list(map(dates_mapping.get, dates_list_))) for dates_list_ in dates_list]
    #initialize figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = [], [], []
    #initialize metric lists
    metric_det_cov_matrix_list, multi_dim_coeff_variation_list = [], []
    #neuron shift
    all_neuron_shift = []
    all_neuron_list = []
    for i, X in enumerate(X_list):
        dates = dates_list[i]
        color_dic = color_dic_list[i]
        if dates_axis == 'z':
            x.append(X[:, 0] + displace[i])
            x_label = labels[0]
            ax.set_xticks([])
            y.append(X[:, 1])
            ax.set_yticks([])
            y_label = labels[1]
            z.append(dates_list[i]+date_correction)
            z_label = 'Mouse age (months)'
            x_scale, y_scale, z_scale = 2, 1, 3
            if include_links:
                ax, metric_det_cov_matrix, multi_dim_coeff_variation, neuron_shift, neuron_list = plot_centroids(X,
                                                                                                    dates_list[i],
                                                                                                    clusters_list[i],
                                                                                                    ax,
                                                                                                    displace[i],
                                                                                                    color_dic_list[i],
                                                                                                    dates_axis='z',
                                                                                                    neuron_list=neurons_list[i],
                                                                                                    date_correction=date_correction,
                                                                                                    verbose=verbose)
                all_neuron_shift.append(neuron_shift)
                all_neuron_list.append(neuron_list)
            ax.set_zticks([])
            ax.set_zlim(zlims)
        elif dates_axis == 'y':
            x.append(X[:, 0] + displace[i])
            x_label = labels[0]
            ax.set_xticks([])
            y.append(dates_list[i]+date_correction)
            y_label = 'Mouse age (months)'
            z.append(X[:, 1])
            z_label = labels[1]
            ax.set_zticks([])
            if include_links:
                ax, metric_det_cov_matrix, multi_dim_coeff_variation, neuron_shift, neuron_list = plot_centroids(X,
                                                                                                                dates_list[i],
                                                                                                                clusters_list[i],
                                                                                                                ax,
                                                                                                                displace[i],
                                                                                                                color_dic_list[i],
                                                                                                                dates_axis='y',
                                                                                                                neuron_list=neurons_list[i],
                                                                                                                date_correction=date_correction,
                                                                                                                verbose=verbose)
                all_neuron_shift.append(neuron_shift)
                all_neuron_shift.append(neuron_shift)
                all_neuron_list.append(neuron_list)
            ax.set_yticks([])
            x_scale, y_scale, z_scale = 2, 3, 1
        elif dates_axis == 'x':
            x.append(dates_list[i]+date_correction)
            x_label = 'Mouse age (months)'
            y.append(X[:, 1])
            y_label = labels[1]
            ax.set_yticks([])
            z.append(X[:, 0] + displace[i])
            z_label = labels[0]
            ax.set_zticks([])
            x_scale, y_scale, z_scale = 3, 1, 2
        #append metrics
        if include_links:
            metric_det_cov_matrix_list.append(metric_det_cov_matrix)
            multi_dim_coeff_variation_list.append(multi_dim_coeff_variation)
        
    x = np.hstack(x)
    y = np.hstack(y)
    z = np.hstack(z)
    clusters = np.hstack(clusters_list)
    #get color list
    color_list = []
    for i, el in enumerate(clusters_list):
        color_list.extend(list(map(color_dic_list[i].get, el.astype('int').astype('str'))))
    if animate:
        ax = set_ax_style(ax, x_label, y_label, z_label, x_scale, y_scale, z_scale)
        def init():
            ax.scatter(x, y, z, edgecolors='k', linewidths=0.5,
              c=color_list)
            return fig, 
        def animate(i):
            ax.view_init(elev=30., azim=i)
            return fig, 
        writergif = animation.PillowWriter(fps=30)
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20, blit=True)
        anim.save(anim_save_path, writer=writergif)
    else:
        ax = set_ax_style(ax, x_label, y_label, z_label, x_scale, y_scale, z_scale)
        if include_points:
            ax.scatter(x, y, z, edgecolor= 'black', s=dot_size, linewidth=lw,
                  c=color_list, zorder=5)   
        ax.azim = azim
        ax.elev = elev
        ax.dist = dist
        #ax.set_xlim(xlim)
        #ax.set_ylim(ylim)
        if savefig:
            plt.savefig(savefig_path, format=file_format, dpi=300)
    plt.show()
    if include_links:
        return np.hstack(metric_det_cov_matrix_list), np.hstack(multi_dim_coeff_variation_list), all_neuron_shift, all_neuron_list
    else:
        return [], []

from sklearn.decomposition import PCA

def pca_channel_timepoint(df, spike_cols,
            prior_cluster_key,
            date_key,
            channel_key,
            new_columns_coord_key,
            cluster_colors, whiten=True, plot=False, figsize=(20, 8)):
    '''
    Function to compute day by day pca representation based off spikes.
    Args:
        df (pandas DataFrame)
        spike_cols (list of str)
        prior_cluster_key (str)
        date_key (str)
        new_columns_coord_key (str)
        cluster_colors (dict)
        whiten (bool): passed to pca, whether to whiten the input
        figsize (tuple)
    Returns:
        df (pandas DataFrame): with added columns
    '''
    if plot:
        fig, axs = plt.subplots(len(df[channel_key].unique()),
                                len(df[date_key].unique()),
                                figsize=figsize)
    df[new_columns_coord_key] = 0
    for i, channel in enumerate(df[channel_key].unique()):
        mask_chan = df[channel_key] == channel
        for j, day in enumerate(df[mask_chan][date_key].unique()):
            mask_date = df[date_key] == day
            mask = mask_chan & mask_date
            spikes_ = df.loc[mask, spike_cols].values
            prior_clusters = df.loc[mask, prior_cluster_key].values
            pca_coordinates = PCA(n_components=2, whiten=whiten).fit_transform(spikes_)
            df.loc[mask, new_columns_coord_key] = pca_coordinates
            if plot:
                axs[i, j].scatter(pca_coordinates[:, 0],
                                    pca_coordinates[:, 1],
                                    c=list(map(cluster_colors.get, prior_clusters)))
                axs[i, j].set_title('Channel-'+str(channel)+'-day-'+str(day))
    fig.tight_layout()
    plt.show()
    return df

def pca_channel(df, spike_cols,
            prior_cluster_key,
            channel_key,
            new_columns_coord_key,
            cluster_colors, whiten=True, plot=False, figsize=(10, 3)):
    '''
    Function to compute day by day pca representation based off spikes.
    Args:
        df (pandas DataFrame)
        spike_cols (list of str)
        prior_cluster_key (str)
        channel_key (str)
        new_columns_coord_key (str)
        cluster_colors (dict)
        whiten (bool): passed to PCA, whether to whiten the input
        figsize (tuple)
    Returns:
        df (pandas DataFrame): with added columns
    '''
    if plot:
        fig, axs = plt.subplots(1, len(df[channel_key].unique()),
                                figsize=figsize)
    df[new_columns_coord_key] = 0
    for i, channel in enumerate(df[channel_key].unique()):
        mask = df[channel_key] == channel
        spikes_ = df.loc[mask, spike_cols].values
        prior_clusters = df.loc[mask, prior_cluster_key].values
        pca_coordinates = PCA(n_components=2, whiten=whiten).fit_transform(spikes_)
        df.loc[mask, new_columns_coord_key] = pca_coordinates
        if plot:
            axs[i].scatter(pca_coordinates[:, 0],
                                pca_coordinates[:, 1],
                                c=list(map(cluster_colors.get, prior_clusters)))
            axs[i].set_title('Channel-'+str(channel)+'overall_days')
    fig.tight_layout()
    plt.show()
    return df