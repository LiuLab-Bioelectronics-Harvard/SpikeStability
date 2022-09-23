from numpy.core.fromnumeric import transpose
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy 

from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, silhouette_score
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from spike_stability.util.calculate_features import features_5
from spike_stability.util.confidence_ellipse import confidence_ellipse
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from scipy import interpolate
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.gaussian_process import GaussianProcessRegressor

def smooth_trajectory(X, dates, cmap, c, ax, gradient=True, lw=4., k=3, s=2):
    #find smooth trajectory using bivariate splines
    tck, u = interpolate.splprep([X[:, 0], X[:, 1], X[:, 2]], s=s, k=k)
    u_fine = np.linspace(0, 1, X.shape[0])
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
    #now plot rhis trajectory with color gradient!
    if gradient:
        for i in range(len(x_fine)-1):
            ax.plot(x_fine[i:i+2], y_fine[i:i+2], z_fine[i:i+2], c=c[i], lw=lw)
            #ax.plot(x_knots[i:i+2], y_knots[i:i+2], z_knots[i:i+2], c=c[i], lw=lw)
    return x_fine, y_fine, z_fine 


def plot_features_mean_trajectory(df,
                                  mouse,
                                  color_maps,
                                  neuron_key,
                                  features=['peak_to_valley',
                                            'peak_trough_ratio',
                                            'repolarization_slope'],
                                  xlabel='peak to valley (ms)',
                                  ylabel='peak trough ratio',
                                  zlabel='repolarization slope',
                                  neurons_plot=None,
                                  trajectory=False,
                                  neuron_trajectory=[i for i in range(9)],
                                  savefig=False,
                                  lw=4.,
                                  k=3,
                                  date_key='date',
                                  mean=False,
                                  s=2,
                                  grid=True,
                                  colorbars=True,
                                  azim=None,
                                  dist=None,
                                  elev=None,
                                  savefigpath=''):
    '''
    Plot trajectory in selected 3D feature space. Trajectories are calculated 
    using b-spline interpolation. 
    Args:
        df (pandas DataFrame): should have columns ['peak_to_valley', 'peak_trough_ratio', 'repolarization_slope', 'recovery_slope', 'amplitude']
        mouse (int): mouse identifier, should be in df['mouse']
        features (list of str): 
        color_maps (dict): mapping from unique neuron label to color e.g. {'0': '#F0000'}
        features (list of str): each str should be a column identifier in df
        neurons_plot (bool): neurons to plot, if None then all neurons are plotted by default
        trajectory (bool): whether to include trajectory in the plot
        neuron_trajectory (list): list of neurons for which to compute and plot the trajectory
        savefig (bool)
        lw (float): linewidth for trajectory path plot
        k (int): bspline order 
        date_key (str): date column identifier in df
        mean (bool): whether to compute mean (if True) or median (if False)
        s (float): dot size
        grid (bool):  whether to include background grid
        colorbars (bool): whether to include colorbars
        azim (float): controll azimuth of plot
        dist (float): control distance of observer to plot
        elev (float)
        savefigpath (str): path to save the figure
    Output:
        trajectory_df (pandas DataFrame): corresponding trajectory
        df_n_mean (pandas DataFrame): mean values for each neuron and feature

    '''
    df_ = df[df['mouse'] == mouse].dropna()
    df_.loc[:, features] = df_[features].apply(pd.to_numeric)
    #alphas = [1 if i in neuron_trajectory else 0.2 for i in range(len(df_[neuron_name].unique()))]
    alphas = [1 for i in range(len(df_[neuron_key].unique()))]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    #to save trajectories
    if trajectory:
        list_x, list_y, list_z, list_neuron = [], [], [], []
    if not neurons_plot:     
        neurons_plot = df_[neuron_key].unique()
    for i, neuron in enumerate(neurons_plot):
        if isinstance(df_[neuron_key].values[0], int):
            df_n = df_[df_[neuron_key] == int(neuron)]
        elif isinstance(df_[neuron_key].values[0], float):
            df_n = df_[df_[neuron_key] == float(neuron)]
        elif isinstance(df_[neuron_key].values[0], str):
            df_n = df_[df_[neuron_key] == str(neuron)]
        if mean:
            df_n_mean = df_n.groupby([date_key]).mean()
        else:
            df_n_mean = df_n.groupby([date_key]).median()  
        X = df_n_mean[features[:3]].values          
        colors = [plt.cm.get_cmap(color_maps[str(neuron)])(((int(el)/len(df_n[date_key].unique()))*(1-1/4))+1/4) for el in df_n_mean.index]
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, s=50, edgecolor='black', linewidth=0.5,
                  depthshade=False)#, alpha=alphas[i])
        #ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=np.array(df_n_mean.index),
         #          cmap=color_maps[str(neuron)],s=50, edgecolor='black', linewidth=0.5)
            
        if trajectory and (int(neuron) in neuron_trajectory):
            print(X.shape)
            x_fine, y_fine, z_fine  = smooth_trajectory(X,
                            df_n_mean.index,
                            plt.cm.get_cmap(color_maps[str(int(neuron))]),
                            colors,
                            ax,
                             lw=lw,
                              s=s,
                             k=k)
            list_x.extend(x_fine)
            list_y.extend(y_fine)
            list_z.extend(z_fine)
            list_neuron.extend([neuron for _ in x_fine])
            
    if grid:
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_zlabel(zlabel, fontsize=15)
    if colorbars:
        bb_anchor2 = [(0.27, -0.2-0.06*i, 1, 1) for i in range(len(df_[neuron_key].unique()))]
        for i, neuron in enumerate(range(len(df_[neuron_key].unique()))):
            cbaxes = inset_axes(ax, width="20%", height="2%", loc="upper right",
                                bbox_to_anchor=bb_anchor2[i], bbox_transform=ax.transAxes)
            norm = matplotlib.colors.Normalize(vmin=5, vmax=18)
            ticks = [5, 18]
            cbar = matplotlib.colorbar.ColorbarBase(cbaxes,
                                                    cmap=plt.cm.get_cmap(color_maps[str(int(neuron))]),
                                                    norm=norm,
                                                    orientation='horizontal',
                                                   ticks=ticks)
            if neuron == len(df_[neuron_key].unique()) - 1:
                cbar.set_label('Age (months)', fontsize=15)
            #cbar.ax.set_yticklabels(ticks, fontsize=10)
    if azim:
        ax.azim = azim
    if elev:
        ax.elev = elev
    if dist:
        ax.dist = dist
    if not grid:
        ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.set_axis_off()
    if savefig:
        plt.savefig(savefigpath, bbox_inches="tight")
    plt.show()
    if trajectory:
        if len(neuron_trajectory) > 0:
            trajectory_df = pd.DataFrame(list_x, columns=['x'])
            trajectory_df['y'] = list_y
            trajectory_df['z'] = list_z
            trajectory_df['neuron'] = list_neuron
            return trajectory_df, df_n_mean
    else:
        return df_n_mean

def median_absolute_deviation(x, features=['peak_to_valley',
                                        'peak_trough_ratio',
                                        'repolarization_slope',
                                        'recovery_slope']):
    for col in features:
        med = np.median(x[col].values)
        x[col] = abs(x[col].values - med)
    return x
    
def feature_correlations_mean(df,
                       mouse,
                       features,
                       color_dic,
                       date_key='',
                       neuron_name='neuron Wavemap',
                       savefig=False, 
                       savefigpath=''):
    '''
    Plot pairplot of features.
    '''
    if isinstance(df['mouse'][0], str):
        df_ = df[df['mouse'] == str(mouse)][features+[neuron_name, date_key]].dropna()
    else:
        df_ = df[df['mouse'] == int(mouse)][features+[neuron_name, date_key]].dropna()
    df_mean = df_.astype(float).groupby([neuron_name, date_key]).mean() 
    df_mean[neuron_name] = [str(el[0]) for el in df_mean.index]
    print(df_mean[features + ['amplitude', neuron_name]])
    sns.set_palette(sns.color_palette(list(color_dic.values())))
    sns.pairplot(data=df_mean[features + [neuron_name]],
                 vars=features,
                 hue=neuron_name)
    if savefig:
        plt.savefig(savefigpath)
    plt.show()
    return df_mean

def colorbars2(ax, d, n, color_maps, position):
    if position == 'upper right':
        bb_anchor = [(-0.01, 0, 1, 1), (-0.01, -0.06, 1, 1)]
    elif position == 'upper left':
        bb_anchor = [(+0.01, 0, 1, 1), (+0.01, -0.06, 1, 1)]
    for i, n_ in enumerate(np.unique(n)):
        cbaxes = inset_axes(ax, width="15%", height="2%", loc=position,
                            bbox_to_anchor=bb_anchor[i], bbox_transform=ax.transAxes)
        color_map = plt.cm.get_cmap(color_maps[str(n_)])
        norm = mpl.colors.Normalize(vmin=np.min(d), vmax=np.max(d))
        ticks = [np.min(d), np.max(d)]
        cb_ = mpl.colorbar.ColorbarBase(cbaxes, cmap=color_map,
                                       norm=norm, orientation='horizontal', ticks=ticks)
        if n_ == np.unique(n)[-1]:
            cb_.set_label('Age (months)')
    
def plot_features_across_time(df,
                            mouse,
                            color_dic,
                            neurons,
                            alpha_list=[],
                            linewidth_list=[],
                            elinewidth_list=[],
                            neuron_LR = [0, 2],
                            neuron_name='neuron Wavemap',
                            figsize=(14, 4),
                            wspace=1.0,
                            date_key='',
                            channel_key='',
                            savefig=False,
                            postprocess=False,
                            features=[],
                            legend=False,
                            median=False,
                            savefigpath=False,
                            scale_amplitude=False,
                            date_offset=0,
                            savecsv=False,
                            mean=False,
                            csvtype='wavemap',
                            savecsvpath='../results/all/'):
    '''
    Function to plot calculated features over time and perform linear regression for chosen neurons.
    Args:
        df (pandas dataframe): dataframe with features to plot
        mouse (int)
        color_dic (dict): color mapping from neuron to color
        neurons (list of int): neurons for which to plot feature evolution
        alpha_list (list of float): list of transparencies with value for each neuron 
        linewidth_list (list of float): list of lw ith value for each each neuron
        elinewidth_list (list of float): errorbar lw
        neuron_LR (list of int): neurons for which to perform Linear Regression
        neuron_name (str): neuron identifier in df
        features (list of str): feature identifiers in df
        spike_cols (list of str): identifiers for spikes in df
        date_offset (int): to offset date axis in plots
        figsize, bbox_to_anchor, wspace -> matplotlib.pyplot plot args
        savefig, savefigpath, savecsv, savecsvpath -> saving params
        csvtype (str): str added to savecsvpath
    Returns:
    '''
    spike_cols = ['t'+str(i) for i in range(1, 31)]
    if isinstance(df['mouse'][0], str):
        df_ = df.copy()
        df_ = df_[df_['mouse'] == str(mouse)][features+[channel_key, neuron_name, date_key]+spike_cols]
        df_ = df_.dropna()
    else:
        df_ = df.copy()
        df_ = df_[df_['mouse'] == int(mouse)][features+[channel_key, neuron_name, date_key]+spike_cols]
        df_ = df_.dropna()
    if scale_amplitude:
        df_.loc[:, 'amplitude'] = df_['amplitude']*1000.
    df_.loc[:, features] = df_[features].apply(pd.to_numeric)
    #initialize figure
    fig, axs = plt.subplots(2, 3, figsize=figsize)
    df_mean = df_.groupby([date_key, neuron_name]).mean()
    df_std = df_.groupby([date_key, neuron_name]).std()
    if median:
        df_median = df_.groupby([date_key, neuron_name]).median()
        df_MAD =  df_.groupby([date_key, neuron_name]).apply(median_absolute_deviation).groupby([date_key, neuron_name]).median()
    d, n = [el[0] for el in df_mean.index], [el[1] for el in df_mean.index]
    if savecsv:
        if postprocess:
            if median:
                df_median.to_csv(savecsvpath+'features_MEDIAN_values_per_timepoint_and_cluster_mouse'+str(mouse)+csvtype+'_postprocess'+'.csv')
                df_MAD.to_csv(savecsvpath+'features_MAD_values_per_timepoint_and_cluster_mouse'+str(mouse)+csvtype+'_postprocess'+'.csv')
            df_mean.to_csv(savecsvpath+'features_MEAN_values_per_timepoint_and_cluster_mouse'+str(mouse)+csvtype+'_postprocess'+'.csv')
            df_std.to_csv(savecsvpath+'features_STDEV_values_per_timepoint_and_cluster_mouse'+str(mouse)+csvtype+'_postprocess'+'.csv')
        else:
            if median:
                df_median.to_csv(savecsvpath+'features_MEDIAN_values_per_timepoint_and_cluster_mouse'+str(mouse)+csvtype+'.csv')
                df_MAD.to_csv(savecsvpath+'features_MAD_values_per_timepoint_and_cluster_mouse'+str(mouse)+csvtype+'.csv')
            df_mean.to_csv(savecsvpath+'features_MEAN_values_per_timepoint_and_cluster_mouse'+str(mouse)+csvtype+'.csv')
            df_std.to_csv(savecsvpath+'features_STDEV_values_per_timepoint_and_cluster_mouse'+str(mouse)+csvtype+'.csv')
    if median:
        df_median['d'], df_median['n'] = d, n
    df_mean['d'], df_mean['n'] = d, n
    list_n = []
    list_r = []
    list_p = []
    list_f = []
    for i, neuron in enumerate(np.unique(n)):
        if int(neuron) in neurons:
            if mean:
                df_median_ = df_mean[df_mean['n'] == neuron]
                df_MAD_ = df_std[df_mean['n'] == neuron]
            elif median:
                df_median_ = df_median[df_median['n'] == neuron]
                df_MAD_ = df_MAD[df_median['n'] == neuron]

            x = np.array([i for i in range(len(df_median_['d'].unique()))])+date_offset
            #first row
            axs[0, 0].errorbar(x,
                            df_median_['peak_to_valley'],
                            df_MAD_['peak_to_valley'],
                            c=color_dic[neuron],
                            ecolor=color_dic[neuron],
                            alpha=alpha_list[i],
                            linewidth=linewidth_list[i],
                            elinewidth=elinewidth_list[i],
                            capsize=5.)
            axs[0, 0].set_ylabel('PT duration (ms)', fontsize=15)
            axs[0, 0].set_ylim(0, 2.5)
            axs[0, 0].set_xlabel('Age (months)', fontsize=15)
            axs[0, 1].errorbar(x,
                            df_median_['repolarization_slope'],
                            df_MAD_['repolarization_slope'],
                            c=color_dic[neuron],
                            ecolor=color_dic[neuron],
                            alpha=alpha_list[i],
                            linewidth=linewidth_list[i],
                            elinewidth=elinewidth_list[i],
                            capsize=5.)
            axs[0, 1].set_ylabel('Repolarization slope', fontsize=15)
            axs[0, 1].set_ylim(0, 1.2)
            axs[0, 1].set_xlabel('Age (months)', fontsize=15)
            #second row
            axs[1, 0].errorbar(x,
                            df_median_['peak_trough_ratio'],
                            df_MAD_['peak_trough_ratio'],
                            c=color_dic[neuron],
                            ecolor=color_dic[neuron],
                            alpha=alpha_list[i],
                            linewidth=linewidth_list[i],
                            elinewidth=elinewidth_list[i],
                            capsize=5.)
            axs[1, 0].set_ylabel('PT ratio', fontsize=15)
            axs[1, 0].set_ylim(0, 1.5)
            axs[1, 0].set_xlabel('Age (months)', fontsize=15)
            axs[1, 1].errorbar(x,
                            df_median_['recovery_slope'],
                            df_MAD_['recovery_slope'],
                            c=color_dic[neuron],
                            ecolor=color_dic[neuron],
                            alpha=alpha_list[i],
                            linewidth=linewidth_list[i],
                            elinewidth=elinewidth_list[i],
                            capsize=5.)
            axs[1, 1].set_ylabel('Recovery slope', fontsize=15)
            axs[1, 1].set_ylim(-0.3, 0.1)
            axs[1, 1].set_xlabel('Age (months)', fontsize=15)
            #third row
            axs[0, 2].errorbar(x,
                            df_median_['amplitude'],
                            df_MAD_['amplitude'],
                            c=color_dic[neuron],
                            ecolor=color_dic[neuron],
                            alpha=alpha_list[i],
                            linewidth=linewidth_list[i],
                            elinewidth=elinewidth_list[i],
                            capsize=5.)
            axs[0, 2].set_ylabel('Amplitude (mv)', fontsize=15)
            axs[0, 2].set_xlabel('Age (months)', fontsize=15)
            axs[0, 2].set_ylim(-0.1, 1.2)
                            
            if int(neuron) in neuron_LR: 
                for j, feat in enumerate(features):
                    if len(x) > 1: 
                        reg = LinearRegression().fit(x.reshape(-1, 1), df_median_[feat].values)
                        slope, intercept = float(reg.coef_), float(reg.intercept_)
                        axs[j%2, j//2].plot(x, reg.predict(x.reshape(-1, 1)),
                                            c=color_dic[neuron],
                                            linestyle='--', label='y={:.4f}x+{:.4f}'.format(slope,intercept), zorder=10)
                        r, p = scipy.stats.pearsonr(x, df_median_[feat].values)
                        list_n.append(neuron)
                        list_f.append(feat)
                        list_r.append(r)
                        list_p.append(p)
    stats_df = pd.DataFrame(np.array([list_n, list_f, list_r, list_p]).T, columns=['neuron', 'feature', 'Pearson r', 'p-value'])
    if legend:
        axs[0, 0].legend(fontsize=12)
        axs[0, 1].legend(fontsize=12)
        axs[1, 0].legend(fontsize=12)
        axs[1, 1].legend(fontsize=12)
        axs[0, 2].legend(fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(wspace=wspace)
    axs[0, 0].grid(False)
    axs[0, 1].grid(False)
    axs[1, 1].grid(False)
    axs[1, 0].grid(False)
    axs[0, 2].grid(False)
    axs[1, 2].grid(False)
    axs[1, 2].axis('off')
    #plt.subplots_adjust(hspace=0.25)
    #fig.tight_layout()
    if savefig:
        plt.savefig(savefigpath)
    plt.show()
    return stats_df



def colorbars(axs, d, n, color_maps):
    for i, n_ in enumerate(np.unique(n)):
        color_map = plt.cm.get_cmap(color_maps[str(n_)])
        norm = mpl.colors.Normalize(vmin=np.min(d), vmax=np.max(d))
        cb_ = mpl.colorbar.ColorbarBase(axs[i+1], cmap=color_map,
                                       norm=norm, orientation='vertical')
        cb_.set_label('Timepoints')


def multi_dim_coeff_variation_VN(centroids):
    cov_matrix = np.cov(centroids.T)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mean = np.mean(centroids, axis=0)
    print(mean)
    return 1/np.sqrt((np.dot(mean.T, np.dot(inv_cov_matrix, mean))))

            
def det_cov_matrix(centroids):
    cov_matrix = np.cov(centroids.T)
    return np.sqrt(np.linalg.det(cov_matrix))

def plot_pc_across_timepoint(df,
                             neurons,
                             mouse,
                             spike_cols,
                             color_dic,
                             neuron_name='neuron Wavemap',
                             color_maps={},
                             confidence=False,
                             savefig=False,
                             mcv=False,
                             date_key='',
                             savefigpath='',
                             legend_position="upper right",
                             figsize=(15, 10)):
    '''
    Plot centroids with st.dev. confidence ellipse of principal component clusters
    for each neuron over time. 
    Args:
        df (pandas dataframe)
        neurons (list of int): neurons for which to plot this
        mouse (int)
        spike_cols (list of str): identifiers for spike values in df
        color_dic (dict): mapping dict for neurons to color
        neuron_name (str): neuron identifier in df
        color_maps (dict): mapping dict for neurons to color map
        confidence (bool): include confidence ellipse
        savefig (bool), savefigpath(str) 
    Returns:
    '''
    if isinstance(df['mouse'][0], str):
        df_ = df[df['mouse'] == str(mouse)]
    else: 
        df_ = df[df['mouse'] == int(mouse)]
    X_ = df_[spike_cols]
    d_, n_ = df_[date_key].values, df_[neuron_name].values
    means, stds, cs = defaultdict(list), defaultdict(list), defaultdict(list)
    fig, ax = plt.subplots(1,
                            figsize=(10,10))
    for date in np.unique(d_):
        mask = d_ == date
        X_date = X_[mask]
        n_date = n_[mask]
        X_pca = PCA(n_components=2, whiten=True).fit_transform(X_date)
        for n in np.unique(n_date):
            if n in neurons:
                mask_n = n_date == n
                X_PCA_n = X_pca[mask_n]
                mean, std = np.mean(X_PCA_n, axis=0), np.std(X_PCA_n, axis=0)
                color = plt.cm.get_cmap(color_maps[str(n)])(int(date)/len(np.unique(d_)))
                ax.scatter(mean[0], mean[1], color=color)
                if confidence:
                    confidence_ellipse(X_PCA_n[:, 0], X_PCA_n[:, 1], ax, 2.0, edgecolor=color)
                means[n].append(mean)
                stds[n].append(std)
                cs[n].append(color_dic[n])
    if mcv:
        for n in neurons:
            print(np.array(means[n]))
            vn_coeff = multi_dim_coeff_variation_VN(np.array(means[n]))
            det_cov_matrix_coeff = det_cov_matrix(np.array(means[n]))
            print(f'Multi dim coefficient of variation for neuron {n} is {vn_coeff}')
            print(f'Sqrt det of covariance matrix for neuron {n} is {det_cov_matrix_coeff}')
    colorbars2(ax, d_, neurons, color_maps, legend_position)
    #means = np.array(means)
    #stds = np.array(stds)
    #plt.scatter(means[:, 0], means[:, 1], c=cs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('PC 1', fontsize=20)
    ax.set_ylabel('PC 2', fontsize=20)
    if savefig:
        plt.savefig(savefigpath)
    plt.show()
    return means, stds