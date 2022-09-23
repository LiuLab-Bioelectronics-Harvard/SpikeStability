import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, product
from spike_stability.util.confidence_ellipse import confidence_ellipse
from spike_stability.util.colors import linear_gradient

def interpolate(centroid_1, centroid_2, nb_points):
    x = np.array([centroid_1[0], centroid_2[0]])
    y = np.array([centroid_1[1], centroid_2[1]])
    t = np.linspace(0, 1, nb_points, endpoint=True)
    xvals = centroid_1[0]*(1-t) + t*centroid_2[0]
    yvals = centroid_1[1]*(1-t) + t*centroid_2[1]
    return xvals, yvals

def interpolation_latent_space(df,
                               mouse, 
                               channel,
                               decoder,
                               cluster_key='',
                               coord_keys=[],
                               spike_cols=['t'+str(i) for i in range(1, 31)],
                               color_dic={},
                               figsize=(12, 4),
                               savefig=False,
                               savefigpath='',
                               nb_points=10):
    #get relevant data
    cols_keep = spike_cols + ['channel', 'dates', 'mouse', cluster_key] + coord_keys
    df_ = df[(df['mouse'] == mouse) & (df['channel'] == channel)][cols_keep]
    clusters = df_[cluster_key].unique()
    cluster_combs = combinations(clusters, 2)
    for i, (c1, c2) in enumerate(cluster_combs):
        #fetch centroids
        cluster1 = df_[df_[cluster_key] == c1][coord_keys].values
        cluster2 = df_[df_[cluster_key] == c2][coord_keys].values
        centroid1 = np.mean(cluster1, axis=0)
        centroid2 = np.mean(cluster2, axis=0)
        #initialize figure
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows=2, ncols=nb_points//2+3)
        #get points between centroids
        xinterp, yinterp = interpolate(centroid1, centroid2, nb_points)
        coord_interp = np.array([[x, y] for x, y in zip(xinterp, yinterp)])
        #get associated waveforms
        wvfs = decoder(coord_interp).numpy()
        #color gradient
        color_dic_grad = tools.linear_gradient(color_dic[str(c1)], color_dic[str(c2)], n=nb_points)
        list_color_grad = color_dic_grad["hex"]
        #plot clouds, centroids and line
        ax1 = fig.add_subplot(gs[:, :3])
        tools.confidence_ellipse(cluster1[:, 0], cluster1[:, 1], ax1, edgecolor=color_dic[str(c1)], n_std=1.0)
        tools.confidence_ellipse(cluster2[:, 0], cluster2[:, 1], ax1, edgecolor=color_dic[str(c2)], n_std=1.0)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xlabel('z1')
        ax1.set_ylabel('z2')
        ax1.scatter(centroid1[0], centroid1[1], marker='o',
                    facecolor=color_dic[str(c1)], edgecolor='black', s=10)
        ax1.scatter(centroid2[0], centroid2[1], marker='o',
                    facecolor=color_dic[str(c2)], edgecolor='black', s=10)
        ax1.scatter(xinterp, yinterp, marker='x', c=list_color_grad)
        ax1.plot(xinterp, yinterp, '--', linewidth=0.3, c='black')
        #plot reconstructed waveforms
        ylim = (np.min(wvfs)+5, np.max(wvfs)+5)
        for k in range(len(wvfs)//2):
            ax2 = fig.add_subplot(gs[0, 3+k])
            ax3 = fig.add_subplot(gs[1, 3+k])
            ax2.set_ylim(ylim)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.set_ylim(ylim)
            ax2.plot(wvfs[k], c=list_color_grad[k])
            ax3.plot(wvfs[len(wvfs)//2+k], c=list_color_grad[len(wvfs)//2+k])
        if savefig:
            plt.savefig(savefigpath+'_'+str(i)+'.pdf', format='pdf')
            plt.savefig(savefigpath+'_'+str(i)+'.jpg', format='jpg')
        plt.show()