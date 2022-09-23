import pandas as pd 
import numpy as np
import pylab
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
import itertools

def color_cm(cmap, NUM_COLORS, transparency=1):
    color = []
    color_idx = 0
    cm = pylab.get_cmap(cmap)
    for i in range(NUM_COLORS):
        cl = list(cm(1. * i / NUM_COLORS))
        cl[-1] = transparency
        color.append(matplotlib.colors.to_hex(tuple(cl)))
    return color

def isi_distribution_plot(df,
                          mouse,
                          color_maps,
                          color_dic,
                          method='waveclus',
                          cluster_key='',
                          savefig_path='',
                          min_spikes=300,
                          nb_bins=30,
                          savefig=False,
                          bar_width=10.,
                          date_key='',
                          correct_date=False,
                          timestamp_key='',
                          zlim=0.3,
                          display=True):
    '''
    Function calculating and plotting ISI distribution plot for individual neurons over time.
    Plots are 3D plots with ISI profiles calculated per day & neuron.
    Args:
        df (pandas dataframe)
        mouse (int)
        color_maps (dict): keys neuron identifiers, values are plt cmaps
        color_dic (dict)
        method (str): to append to savefile path
        cluster_key: identifier of clusters to consider in df
        savefig_path (str)
        min_spikes (int): threshold below which we don't plot the ISI for that day
        nb_bins (int): nb of bins to divide 500 ms in 
        savefig (bool): whether to save figures
        bar_width (float)
        correct_date (bool): whether to correct date labels directly on the plot
        zlim (float): z limit (percentage) for all ISI plots
        display (bool): whether to show the images
    Returns:
        ISI_distribution (dict): keys=cluster identifiers, values=list of ISI distribution for
        each day
    '''
    df_ = df[(df['mouse'] == mouse)]
    clusters = df_[cluster_key].values
    dates = df_[date_key].values
    timestamps = df_[timestamp_key].values
    
    ISI_distributions = []
    cluster_distributions = []
    stage_distributions = []
    ISI_distribution = {}

    for i, cluster_ in enumerate(np.unique(clusters)):
        #masks
        mask_cluster = clusters == cluster_
        ISI_distribution[cluster_] = []
        #figure 
        if display:
            fig = plt.figure(figsize=(20, 10))
            ax_ = fig.add_subplot(111, projection='3d')    
        
        stages_this_cluster = np.array(dates)[mask_cluster]
        #compute color gradient that respects predefined colormaps based on number of days

        for j, stage_ in enumerate(np.unique(stages_this_cluster)):
            position_in_all_stages = np.unique(dates).tolist().index(stage_) + 1
            #masks
            mask_stage = np.array(dates) == stage_
            mask_total = mask_stage & mask_cluster
            
            if np.sum(mask_total) >= min_spikes:
                #times
                timestamps_ = timestamps[mask_total]  #already in ms
                timestamps_ = sorted(timestamps_)
                #ISIs
                ISIs = np.diff(timestamps_)
                #append
                ISI_distribution[cluster_].append(ISIs)
                ISI_distributions.append(ISIs)
                cluster_distributions.append([cluster_ for _ in ISIs])
                stage_distributions.append([stage_ for _ in ISIs])

                #plotting
                times_ = np.linspace(0, 500, nb_bins)
                counts_, bins_ = np.histogram(ISIs, bins=times_)
                if display:
                    ax_.bar(bins_[:-1],
                        counts_/np.sum(counts_),
                        width=bar_width,
                        zs=position_in_all_stages,
                        zdir='y',
                        align='edge',
                        color=color_dic[cluster_],
                        edgecolor=color_dic[cluster_])
                
        #finalize figure
        if display:
            ax_.grid(False)
            ax_.set_xlabel('Interspike interval (ms)', labelpad=20)
            ax_.set_zlabel('Frequency (%)', labelpad=30)
            ax_.set_zlim(0, zlim)
            if correct_date:
                ax_.set_yticklabels([str(i) for i in np.arange(4, 18, 2)])
            ax_.set_ylabel('Age (months)', labelpad=20)
            #ax_.
            ax_.elev = 20

            ax_view = fig.gca()
            ax_view.get_proj = lambda: np.dot(Axes3D.get_proj(ax_view), np.diag([0.5, 1.0, 0.5, 1]))
            

            if savefig:
                plt.savefig(savefig_path+'/'+method+'_mouse_'+str(mouse)+'_isi_distribution_cluster'+str(cluster_)+'.pdf', format='pdf')
            plt.show()
    ISI_distributions = np.hstack(ISI_distributions)
    cluster_distributions = np.hstack(cluster_distributions)
    stage_distributions = np.hstack(stage_distributions)
    df_isi = pd.DataFrame(ISI_distributions, columns=['isi'])
    df_isi[cluster_key] = cluster_distributions
    df_isi[date_key] = stage_distributions
    return ISI_distribution, df_isi


#fitting exponential decay
def exp_curve_fitting(clusters,
                      isi_distribution,
                      nb_bins,
                      bounds=(),
                      p0=(1, 1, 0.05),
                      upper_thresh=1,
                      lower_thresh=0):
    '''
    Fit an exponential decay to the given distributions. 
    We could fit a more complex gamma-exponential mixture but the simple case should suffice here.
    Args:
        isi_distribution (dict): obtained by isi_distribution_plot function
        nb_bins (int)
        bound (tuple): for scipy.optimize curve fit function
        p0 (tuple): initial parameters for curve fitting function
        upper_thresh (float): values beyond which we consider the fitting results to be wrong or
        distribution profile for that day to be too bad
        lower_thresh (float)
    Returns:
        firing_param_exp_decay (dict):firing parameters obtained by exponential fitting for neurons
    '''
    def monoExp(x, m, t):
        return m * np.exp(-t * x)

    sampleRate = 1000.
    firing_param_exp_decay = {}
    firing_param_exp_decay_std = {}
    #get all combinations of initialization for curve fitting possibilities
    init_values = [[a, b] for a in p0[0] for b in p0[1]]
    for cluster in np.unique(clusters):
        firing_param_exp_decay[cluster] = []
        firing_param_exp_decay_std[cluster] = []
        isi_cluster = isi_distribution[cluster]
        random_date = np.random.randint(len(isi_cluster))
        for i in range(len(isi_cluster)):
            isi_curve = isi_cluster[i]
            times_ = np.linspace(0, 500, nb_bins)
            counts_, bins_ = np.histogram(isi_curve, bins=times_, density=True)
            counts_ = counts_/np.sum(counts_)
            #max_ind = np.argmax(counts_)
            max_ind = 0
            for p0_ in init_values:
                t_list = []
                try:
                    params, cv = curve_fit(monoExp, bins_[:-1][max_ind:], counts_[max_ind:], p0_, bounds=bounds)
                    m, t = params
                    #only taking values within a certain range as exp fitting can yield a bad value
                    #if not converge properly
                    if t <= upper_thresh:
                        if t >= lower_thresh:
                            t_list.append(t)
                    if i == random_date and p0_ == p0[0]:
                        plt.plot(bins_[:-1][max_ind:], counts_[max_ind:])
                        plt.plot(bins_[:-1][max_ind:], monoExp(bins_[:-1][max_ind:], m, t), label='fitted curve')
                        plt.xlabel('')
                        plt.ylabel('freq')
                        plt.legend()
                        plt.show()              
                except Exception as e:
                    print(f'Fit failed for {cluster}')  
            firing_param_exp_decay[cluster].append(np.mean(t_list))
            firing_param_exp_decay_std[cluster].append(np.std(t_list))
    return firing_param_exp_decay, firing_param_exp_decay_std



def isi_distribution_fit(df,
                          mouse,
                          neuron_key,
                          neuron_color_dic,
                          neuron_color_maps,
                          savefig=False,
                          savefig_path_isi='',
                          savefig_path_firing_param='',
                          file_format='.pdf',
                          min_spikes=0,
                          nb_bins=5,
                          display=False,
                          method='Wavemap',
                          bar_width=10.,
                          zlim=0.3,
                          nb_bins_exp=5,
                          p0=[[], []],
                          bounds=(0, [1., 0.2]),
                          lower_thresh=0,
                          upper_thresh=0.1,
                          **kwargs):
    '''
    Perform both plotting of individual ISI profiles as well as fitting the curves and 
    plotting overall boxplot of fitting results.
    Args:
        df (pandas dataframe): contains all relevant data (timestamps, cluster labels, ...)
        mouse (int): mouse indentifier in df dataframe
        neuron_key (str): column with labels for neurons in df
        neuron_color_dic (dict): mapping dict from neuron label to associated color
        neuron_color_maps (dict): mapping dict from neuron label to color map
        savefig (bool): concerns all figures generated by this function
        savevig_path_isi (str): for individual isi plots
        savefig_path_firing_param (str): for boxplot with overall fitting information
        file_format (str)
        min_spikes (int): threshold for considering isi distribution profile for neuron on given day
        nb_bins (int)
        display (bool): whether to display individual isi profiles
        method (str): for figure saving, appended to str path
        bar_width (float): for individual isi plotting
        zlim (float): for individual isi plotting
        **kwargs: passed to exp_curve_fitting function
    Returns:
        isi_distrib (dict): isi distribution profiles per neuron
        df_isi (pandas dataframe): firing parameters for neurons
    '''
    #0.5, 0.001, 0.0 works well for fits
    #get isi distribution
    isi_distrib, df_isi_distrib = isi_distribution_plot(df,
                                        mouse,
                                        neuron_color_maps,
                                        neuron_color_dic,
                                        cluster_key=neuron_key,
                                        savefig_path=savefig_path_isi,
                                        file_format=file_format,
                                        min_spikes=min_spikes,
                                        nb_bins=nb_bins,
                                        display=display,
                                        savefig=savefig,
                                        method=method,
                                        bar_width=bar_width,
                                        zlim=zlim)


    mask = (df['mouse'] == mouse)
    current_clusters = df[mask][neuron_key].values
    #calculate exponential fitting parameter on the distributions
    firing_param_exp_decay, firing_param_exp_decay_std = exp_curve_fitting(current_clusters,
                                               isi_distrib,
                                               nb_bins=nb_bins_exp,
                                               p0=p0,
                                               bounds=bounds,
                                               lower_thresh=lower_thresh,
                                               upper_thresh=upper_thresh) 

    #conver to df
    list_n = np.hstack([np.ones(len(val))*int(key) for key, val in firing_param_exp_decay.items()])
    list_fp = np.hstack(firing_param_exp_decay.values())
    list_fp_std = np.hstack(firing_param_exp_decay_std.values())
    df_isi = pd.DataFrame(list_fp*1000., columns=['firing_param'])
    df_isi[neuron_key] = np.array(list_n).astype('int')
    df_isi['firing_param_std'] = list_fp_std*1000.
    
    #plot
    ax = sns.swarmplot(data=df_isi,
              x=neuron_key,
              y="firing_param", color="0.2")

    ax = sns.boxplot(data=df_isi,
                  x=neuron_key,
                  y="firing_param",
                    palette=dict(list((int(key), val) for key, val in neuron_color_dic.items())))

    #legend
    custom_dots = [Line2D([0], [0], marker='s', color='w', label='neuron '+str(int(i)),
                      markerfacecolor=neuron_color_dic[str(int(i))],
                      markersize=8) for i in df_isi[neuron_key].unique()]

    ax.legend(handles=custom_dots, loc="upper right", bbox_to_anchor=(1.27, 1.03))
    ax.set_xlabel('neuron')
    ax.set_ylabel(r'$\lambda$ firing parameter, $a*e^{-\lambda t}$')
    if savefig:
        plt.savefig(savefig_path_firing_param+'.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(savefig_path_firing_param+'.jpg', format='jpg', bbox_inches='tight')
    plt.show()
    
    #plot 2
    fig, ax = plt.subplots()
    for neuron in df_isi[neuron_key].unique():
        df_isi_ = df_isi[df_isi[neuron_key] == neuron]
        x = np.arange(5, 19)[:len(df_isi_)]
        ax.errorbar(x, df_isi_["firing_param"], df_isi_["firing_param_std"], c=neuron_color_dic[neuron],
                label='neuron '+neuron, marker='x')
    ax.set_xlabel('Age (months)')
    ax.set_ylabel(r'$\lambda$ firing parameter, $a*e^{-\lambda t}$')
    ax.set_ylim(0, 150)
    plt.legend(bbox_to_anchor=(1, 1))
    if savefig:
        plt.savefig('../results/mouse1/firing_param_over_time.jpg')
        plt.savefig('../results/mouse1/firing_param_over_time.pdf')
    plt.show()
    return isi_distrib, df_isi