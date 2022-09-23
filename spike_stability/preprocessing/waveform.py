import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def remove_artefacts(df, spike_cols, threshold, mode='min'):
    '''
    Remove rows of the dataframe containing artifact spikes with values above or below
    a certain threshold.
    Args:
        df (pandas DataFrame): must have spike_cols 
        spike_cols (list): list of str with column identifiers of spike values in df
        threshold (float): threshold in microvolt
        mode (str): either 'min' or 'max' to denote negative or positive spiking
    Returns:
        df (pandas DataFrame): removed rows
    '''
    if mode == "min":
        mask = df[spike_cols].min(axis=1).abs() > threshold
    elif mode == "max":
        mask = df[spike_cols].max(axis=1).abs() > threshold
    df = df.loc[~mask, :]
    return df

def ensure_spike_alignment(df, spike_cols, nb_values_keep, mode='min', plot=False):
    '''
    Keep only those spikes with sensible peak positions. This removes
    the spikes whose alignment was not carried out correctly.
    Args:
        df (pandas DataFrame): must have spike_cols 
        spike_cols (list): list of str with column identifiers of spike values in df
        nb_values_keep (int): number of top spike peak position counts to keep in df.
           A value equal to the spike length (e.g. 30) will keep all spikes. A value of 1 will only
           keep the spikes whose peak position is equal to the majority peak position
        mode (str): positive or negative spikes
        plot (bool): Whether to plot the spike peak distribution histogram
    '''
    if mode == "min":
        peak_positions = np.abs(df[spike_cols].values[:, :len(spike_cols)//2]).argmax(axis=1)
    elif mode == "max":
        peak_positions = df[spike_cols].values.argmax(axis=1)
    values, counts = np.unique(peak_positions, return_counts=True)
    align_idx = np.in1d(peak_positions, values[counts.argsort()[-nb_values_keep:][::-1]])
    print('Keeping only spikes with peak positions in positions: '+str(values[counts.argsort()[-nb_values_keep:][::-1]]))
    #plotting
    if plot:
        plt.figure()
        plt.bar(values, counts)
        plt.title('Spike peak distribution')
    #keep relevant part of df
    df = df.loc[align_idx, :]
    return df