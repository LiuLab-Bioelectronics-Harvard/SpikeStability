import os
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.io import loadmat 
from datetime import datetime
from sklearn.metrics import mean_squared_error
from itertools import permutations 

def mat_datafile_folder_channel(datafile,
                                reg_expression_channel,
                                reg_expression_date,
                                spikes_key='spikes',
                                clusters_timestamps_key='cluster_class'):
    '''
    Short function to extract the useful data from .mat
    files produced by waveclus.
    For .mat file names with xxxx_CHyy_z.mat format, the reg expressions are 
    "CH(.+?)_" and "_(.+?).mat" for channel and date, respectively.
    Args:
        datafile (str): path to datafile
        reg_expression_channel (str): python regular expression to select channel information from datafile name
        reg_expression_date (str): python regular expression to select date information from datafile name
        spikes_key (str): key to mat dictionary for the spiking data
        clusters_timestamps_key (str): key to mat dictionary for timestamp and cluster id data
    Output:
        spikes (2D array)
        clusters (1D array)
        timestamps (1D array)
        date (str)
        channel (str)
    '''
    data = loadmat(datafile)
    spikes = data[spikes_key]
    clusters = data[clusters_timestamps_key][:, 0]
    timestamps = data[clusters_timestamps_key][:, 1]
    try:
        date = re.search(reg_expression_date, datafile.split('/')[-1]).group(1)
        channel = re.search(reg_expression_channel, datafile.split('/')[-1]).group(1)
    except AttributeError:
        print(f"No element found in original string mathing re {reg_expression_channel} or {reg_expression_date} \
            in str {datafile.split('/')[1]}!! ")
    return spikes, clusters.astype('int'), timestamps, date, channel


def mat_datafile_folder_date(datafile,
                            file_folder_name,
                            reg_expression_channel,
                            spikes_key='spikes',
                            clusters_timestamps_key='cluster_class'):
    '''
    Short function to extract the useful data from .mat
    files produced by waveclus. 
    Overall data structure should be file folders with names as %m%d%y
    each containting .mat files for all the individual channels.
    For .mat file names with xxxx_CHyy_z.mat format, the reg expressions are 
    "CH(.+?)_" and "_(.+?).mat" for channel and date, respectively.
    Args:
        datafile(str): path to datafile
        file_folder_name (str): %m%d%y format (e.g 082520)
    Returns:
        spikes (2D array)
        clusters (1D array)
        timestamps (1D array)
        date (str): %Y-%m-%d
        channel (str)
    '''
    data = loadmat(datafile)
    spikes = data[spikes_key]
    clusters = data[clusters_timestamps_key][:, 0]
    timestamps = data[clusters_timestamps_key][:, 1]
    try:
        date = datetime.strptime(file_folder_name, '%m%d%y').strftime('%Y-%m-%d')
        channel = re.search(reg_expression_channel, datafile.split('/')[-1]).group(1)
    except Exception as e:
        print(f'Not able to fetch data for data file {datafile} due to {e}')
    return spikes, clusters.astype('int'), timestamps, date, channel

def give_correct_files(files,
                    file_format='.mat',
                    prefix='times_'):
    '''
    Function that returns all files withing the file folder 
    that have the correct format.
    Args:
        files (list of str)
        file_format (str)
        prefix (str): desired prefix to have on all files
    Returns:
        correct_files (list of str): names of files to include in analysis, sorted by name
    '''
    correct_files = []
    for file_name in files:
        if file_name.endswith(file_format):
            if file_name.startswith(prefix):
                correct_files.append(file_name)
    return correct_files 

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


def plot_waveform_means(spikes,
                        clusters,
                        dates,
                        cluster_colors):
    '''
    Function to plot waveform means of a channel across timepoints.
    '''
    fig, axs = plt.subplots(len(np.unique(dates)), len(np.unique(clusters)),
                            figsize=(10*len(np.unique(clusters)), 10*len(np.unique(dates))),
                            sharex=True, sharey=True)
    t = np.arange(1, spikes.shape[1]+1)
    for i, date_ in enumerate(np.unique(dates)):
        mask1 = dates == date_
        for j, cluster_ in enumerate(np.unique(clusters)):
            mask2 = clusters == cluster_
            mask = mask1 & mask2
            if np.sum(mask) > 0:
                if len(np.unique(clusters)) > 1:
                    axs[i, j].plot(t, np.mean(spikes[mask], axis=0), c=cluster_colors[int(cluster_)], lw=5.)
                    axs[i, j].set_xticks([])
                else:
                    axs[i].plot(t, np.mean(spikes[mask], axis=0), c=cluster_colors[int(cluster_)], lw=5.)
                    axs[i].set_xticks([])
    fig.tight_layout()
    plt.show()

def mat_data_folder_channel(data_basepath,
                            cluster_colors,
                            prefix='times_CH13',
                            file_format='.mat',
                            date_reg_expression='',
                            channel_reg_expression='',
                            exclude_clusters=[0],
                            custom_extraction=False,
                            custom_func=None,
                            spikes_key='spikes',
                            clusters_timestamps_key='cluster_class',
                            verbose=0,
                            **kwargs):
    """
    Automate fetching data from .mat files.
    File folder should have all .mat files inside you wish to use inside.
    Example: all_data/times_CH13_1.mat all_data/times_CH13_2.mat ...
    Args:
        data_basepath (str): base folder containing data
        cluster_colors (dict): mapping dic from neuron cluster id to color
        prefix (str): if you only wish to select files with a certain prefix, e.g. times_
        file_format (str): .mat, loadmat is used to load files
        reg_expression_channel (str): python regular expression to select channel information from datafile name
        reg_expression_date (str): python regular expression to select date information from datafile name
        exclude_clusters (int): cluster labels of waveclus to exclude, e.g. 0 for noise cluster
        custom_extraction (bool): whether to use custom function to read .mat data
        custom_func (function): used to read .mat data if custom_extraction is True
        spikes_key (str): passed to mat_datafile_folder_channel
        clusters_timestamps_key (str): passed to mat_datafile_folder_channel
        verbose (int): verbosity level
        **kwargs: any additional parameters to pass to custom_func other than datafile path.
    Returns:
        spikes (2D array): num_spikes x length of spike (e.g 30 points) 
        channels (1D array): len is num_spikes, containing corresponding channel for each spike
        timestamps (1D array): len is num_spikes, timestamp for each spike
        dates (1D array): len is num_spikes, date for each spike
        clusters (1D array): len is num_spikes, waveclus assigned cluster for each spike
    """
    #init printing function
    if verbose:
        def verboseprint(string, *args, **kwargs):
            print(string, *args, **kwargs)
    else:
        verboseprint = lambda *a, **k: None
    #initialize
    spikes = []
    channels = []
    timestamps = []
    dates = []
    clusters = []
    #go through files
    for dirpath, dirnames, files in os.walk(data_basepath):
        correct_files = give_correct_files(files, file_format=file_format, prefix=prefix)
        for i, file_ in enumerate(sorted(correct_files,  key=lambda x:int(re.search(date_reg_expression, x).group(1)))):
            path_ = dirpath + '/' + file_
            #extract information from file
            if custom_extraction:
                spikes_, clusters_, timestamps_, date_, channel_ = custom_func(path_, **kwargs)
            else:
                spikes_, clusters_, timestamps_, date_, channel_ = mat_datafile_folder_channel(path_,
                                                                                               channel_reg_expression,
                                                                                               date_reg_expression,
                                                                                               spikes_key=spikes_key,
                                                                                               clusters_timestamps_key=clusters_timestamps_key)
            verboseprint(f'Extracted data from {file_} recorded on month {date_}')
            #clusters that we want
            clusters_present = np.unique(clusters_)[~np.isin(np.unique(clusters_).astype('int'), exclude_clusters)]
            verboseprint(f'There are {len(np.unique(clusters_present))} clusters in this file.')
            for j, cluster_ in enumerate(clusters_present):
                #get data for this cluster
                mask_cluster = clusters_ == cluster_
                spikes_cluster = spikes_[mask_cluster]
                timestamps_cluster = timestamps_[mask_cluster] 
                spikes.append(spikes_cluster)
                dates.append(np.array([date_ for _ in range(spikes_cluster.shape[0])]))
                timestamps.append(timestamps_cluster)
                channels.append(np.array([channel_ for _ in range(spikes_cluster.shape[0])]))
                clusters.append(np.ones(len(spikes_cluster))*int(int(cluster_)))
       
    spikes = np.vstack(spikes)
    channels = np.hstack(channels)
    timestamps = np.hstack(timestamps)
    dates = np.hstack(dates)
    clusters = np.hstack(clusters) 
    #plot
    #figure for this channel over all weeks
    plot_waveform_means(spikes,
                        clusters,
                        dates,
                        cluster_colors)
    return spikes, channels, timestamps, dates, clusters


def mat_data_folder_date(data_basepath,
                         cluster_colors,
                         prefix='',
                         file_format='.mat',
                         channel_reg_expression='',
                         exclude_days=[],
                         exclude_clusters=[0],
                         custom_extraction=False,
                         custom_func=None,
                         spikes_key='spikes',
                         clusters_timestamps_key='cluster_class',
                         verbose=0,
                         **kwargs):
    """
    Automate fetching data from .mat files.
    File folder should have individual folders for each 
    recording day. File names for these recordings should be 
    date in %m%d%y e.g. 052220. 
    Args:
        data_basepath (str): base folder containing data
        prefix (str): if you only wish to select files with a certain prefix, e.g. times_
        file_format (str): .mat, loadmat is used to load files
        reg_expression_channel (str): python regular expression to select channel information from datafile name
        exclude_days (list of str): list with folder names to exclude in %m%d%y format
        exclude_clusters (int): cluster labels of waveclus to exclude, e.g. 0 for noise cluster
        custom_extraction (bool): whether to use custom function to read .mat data
        custom_func (function): used to read .mat data if custom_extraction is True
        verbose (int): verbosity level
        **kwargs: any additional parameters to pass to custom_func other than str to datafile
    Returns:
        spikes (2D array): num_spikes x length of spike (e.g 30 points) 
        channels (1D array): len is num_spikes, containing corresponding channel for each spike
        timestamps (1D array): len is num_spikes, timestamp for each spike
        dates (1D array): len is num_spikes, date for each spike
        clusters (1D array): len is num_spikes, waveclus assigned cluster for each spike
    """
    #init printing function
    if verbose:
        def verboseprint(string, *args, **kwargs):
            print(string, *args, **kwargs)
    else:
        verboseprint = lambda *a, **k: None
    #initialize
    spikes = []
    channels = []
    timestamps = []
    dates = []
    clusters = []
    
    #go through files
    for dirpath, dirnames, files in os.walk(data_basepath):
        correct_files = give_correct_files(files, file_format=file_format, prefix=prefix)
        file_folder_name = dirpath.split('/')[-1][:6]
        if not file_folder_name in exclude_days:
            for i, file_ in enumerate(sorted(correct_files)):
                path_ = dirpath + '/' + file_
                #extract information from file
                if custom_extraction:
                    spikes_, clusters_, timestamps_, date_, channel_ = custom_func(path_, file_folder_name, **kwargs)
                else:
                    spikes_, clusters_, timestamps_, date_, channel_ = mat_datafile_folder_date(path_,
                                                                                                file_folder_name,
                                                                                                channel_reg_expression,
                                                                                                spikes_key=spikes_key,
                                                                                                clusters_timestamps_key=clusters_timestamps_key)
                                                                                                
                verboseprint(f'Extracted data from {file_} recorded on timepoint {date_}')
                #clusters that we want
                clusters_present = np.unique(clusters_)[~np.isin(np.unique(clusters_).astype('int'), exclude_clusters)]
                verboseprint(f'There are {len(np.unique(clusters_present))} clusters in this file.')
                for j, cluster_ in enumerate(clusters_present):
                    #get data for this cluster
                    mask_cluster = clusters_ == cluster_
                    spikes_cluster = spikes_[mask_cluster]
                    timestamps_cluster = timestamps_[mask_cluster] 
                    spikes.append(spikes_cluster)
                    dates.append(np.array([date_ for _ in range(spikes_cluster.shape[0])]))
                    timestamps.append(timestamps_cluster)
                    channels.append(np.array([channel_ for _ in range(spikes_cluster.shape[0])]))
                    clusters.append(np.ones(len(spikes_cluster))*int(int(cluster_)))

    spikes = np.vstack(spikes)
    channels = np.hstack(channels)
    timestamps = np.hstack(timestamps)
    dates = np.hstack(dates)
    clusters = np.hstack(clusters) 
    #plot
    #figure for this channel over all weeks
    #initialize figure
    plot_waveform_means(spikes,
                        clusters,
                        dates,
                        cluster_colors)
    return spikes, channels, timestamps, dates, clusters


def mat_data_all_dates(data_basepath,
                        prefix_channel=[],
                        cluster_colors_channel=[],
                        file_format='',
                        channel_reg_expression='',
                        exclude_clusters_channel=[[0]],
                        spikes_key='spikes',
                        clusters_timestamps_key='cluster_class',
                        verbose=0,
                        **kwargs):
    #init printing function
    if verbose:
        def verboseprint(string, *args, **kwargs):
            print(string, *args, **kwargs)
    else:
        verboseprint = lambda *a, **k: None
    all_spikes, all_channels, all_timestamps, all_dates, all_clusters = [], [], [], [], []
    for i, prefix in enumerate(prefix_channel):
        spikes, channels, timestamps, dates, clusters = mat_data_folder_date(data_basepath,
                                                                            cluster_colors_channel[i],
                                                                            prefix=prefix,
                                                                            file_format=file_format,
                                                                            channel_reg_expression=channel_reg_expression,
                                                                            exclude_clusters=exclude_clusters_channel[i],
                                                                            spikes_key=spikes_key,
                                                                            clusters_timestamps_key=clusters_timestamps_key,
                                                                            verbose=verbose,
                                                                            **kwargs)
        all_spikes.append(spikes)
        all_channels.append(channels)
        all_timestamps.append(timestamps)
        all_dates.append(dates)
        all_clusters.append(clusters)

    all_spikes = np.vstack(all_spikes)
    all_channels = np.hstack(all_channels)
    all_timestamps = np.hstack(all_timestamps)
    all_dates = np.hstack(all_dates)
    all_clusters = np.hstack(all_clusters)

    return all_spikes, all_channels, all_timestamps, all_dates, all_clusters



def mat_data_all_channels(data_basepath,
                          channel_paths,
                          cluster_colors_channel,
                          prefix_channel='times',
                          file_format='',
                          date_reg_expression='',
                          channel_reg_expression='',
                          exclude_clusters_channel=[[0]],
                          spikes_key='spikes',
                          clusters_timestamps_key='cluster_class',
                          verbose=0,
                          **kwargs):
    '''
    Fetch data from .mat files (waveclus output) including spikes, channel, timestamps.
    Data should be separate in individual folders per channel.
    e.g data_basepath = './user/project/data/' and folders for channels are 'CH1', 'CH2', ...
    Args:
        data_basepath (str): path to directory containing individual channel folders
        channel_paths (list of str): relative path from data_basepath to channel information
        cluster_colors_channel (list of dict): color mapping dict for each cluster
        prefix_channel (list of str): all prefixes
        file_format (str)
        date_reg_expression (str): passed to mat_data_folder_channel func
        channel_reg_expression (str): passed to mat_data_folder_channel func
        exclude_clusters_channel (list of list of int): clusters to exclude for each channel
        spikes_key (str): key to spiking data in matlab dict obtained from .mat files
        clusters_timestamps_key (str): key to cluster label and timestamp obtained from .mat files
        verbose (int): verbosity level
        **kwargs: args to pass to mat_data_folder_channel
    Returns:
        spikes (2D array): num_spikes x length of spike (e.g 30 points) 
        channels (1D array): len is num_spikes, containing corresponding channel for each spike
        timestamps (1D array): len is num_spikes, timestamp for each spike
        dates (1D array): len is num_spikes, date for each spike
        clusters (1D array): len is num_spikes, waveclus assigned cluster for each spike
    '''
    #init printing function
    if verbose:
        def verboseprint(string, *args, **kwargs):
            print(string, *args, **kwargs)
    else:
        verboseprint = lambda *a, **k: None
    all_spikes, all_channels, all_timestamps, all_dates, all_clusters = [], [], [], [], []
    for i, channel_path in enumerate(channel_paths):
        current_path = data_basepath + channel_path
        verboseprint(f'Processing channel {channel_path} \n')
        channel_path_number = ''.join(x for x in channel_path if x.isdigit())
        spikes, channels, timestamps, dates, clusters = mat_data_folder_channel(current_path,
                                                                            cluster_colors_channel[channel_path_number],
                                                                            prefix=prefix_channel[i],
                                                                            file_format=file_format,
                                                                            date_reg_expression=date_reg_expression,
                                                                            channel_reg_expression=channel_reg_expression,
                                                                            exclude_clusters=exclude_clusters_channel[i],
                                                                            spikes_key=spikes_key,
                                                                            clusters_timestamps_key=clusters_timestamps_key,
                                                                            verbose=verbose,
                                                                            **kwargs)
        all_spikes.append(spikes)
        all_channels.append(channels)
        all_timestamps.append(timestamps)
        all_dates.append(dates)
        all_clusters.append(clusters)

    all_spikes = np.vstack(all_spikes)
    all_channels = np.hstack(all_channels)
    all_timestamps = np.hstack(all_timestamps)
    all_dates = np.hstack(all_dates)
    all_clusters = np.hstack(all_clusters)

    return all_spikes, all_channels, all_timestamps, all_dates, all_clusters


def align_labels(df_, spike_cols, cluster_key, channel_key, date_key,
                 alignment_dic):
    '''
    Various spike sorting algorithms such as waveclus produce different label numbers for the same
    neuron across recording days. As such, it is necessary to align these labels so that a single
    neuron has the same label over time. To do so we minimize the mse of potential pairings between
    average waveforms of a given day and a template day.
    Args:
        df_ (pandas DataFrame)
        spike_cols (list of str)
        cluster_key (str): column identifier in df of cluster labels
        channel_key (str): column identifier of channel
        date_key (str): column identifier of dates
        alignment_dic (dict): dict with channel numbers as keys and days to take as template as values
        all_color_dics (dict): dict of color dics with keys as channelXXmouseY.
        mouse (int): mouse identifier
        color_key (str): str identifier for color code in all_color_dics['channelXXmouseY'] dict.
    Returns:
        df with aligned labels, cluster key column is modified in place
    '''
    df = df_.copy()
    for i, channel in enumerate(df[channel_key].unique()):
        mask_channel = df[channel_key] == channel
        day_align = alignment_dic[channel]
        #get template waveforms for automatic alignment
        df_align = df.loc[df[date_key] == day_align, :]
        templates = df_align.groupby([cluster_key]).mean()[spike_cols].values

        for date in df[date_key].unique():
            mask_date = df[date_key] == date
            mask = mask_channel & mask_date
            spikes_ = df.loc[mask, spike_cols].values
            clusters_ = df.loc[mask, cluster_key].values
            #assignment
            candidates = []
            for cluster in np.unique(clusters_):
                mask_cluster = clusters_ == cluster
                spikes_cluster = spikes_[mask_cluster]
                candidates.append(np.mean(spikes_cluster, axis=0))
            optimal_match = auto_match(templates, candidates)
            #build mapping dic
            mapping_dic = {}
            for pair in optimal_match:
                mapping_dic[int(pair[1])] = int(pair[0])
            print(f'Optimal match found for day {date}: {mapping_dic}, channel {channel}')
            df.loc[mask, cluster_key] = df.loc[mask, cluster_key].apply(lambda x: mapping_dic[x])
    df[cluster_key] = df[cluster_key].astype('int')
    return df


def build_alignment_dic(df, channel_key, date_key, cluster_key):
    '''
    Input dataframe with relevent columns (spikes, cluster labels, dates, channel)
    Returns:
        alignment_dict (dict): channel to day mapping for alignment.
        By default, the first day with max number of clusters is chosen so that no
        cluster label is dropped in the analysis.
    '''
    alignment_dic = {}
    grouped_nb_labels = df.groupby([channel_key, date_key])[cluster_key].unique()
    for el in list(grouped_nb_labels.index):
        chan, date = el[0], el[1]
        if len(grouped_nb_labels[el]) == len(df[(df[channel_key] == chan)][cluster_key].unique()):
            if not (chan in list(alignment_dic.keys())):
                alignment_dic[chan] = date
    return alignment_dic