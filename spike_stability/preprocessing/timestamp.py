import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def remove_bad_timing(df, date_key,
                      timestamp_key, neuron_key,
                      threshold, sampling_rate):
    ''' 
    Removes all spikes for each putative neuron whose consecutive timestamps
    follow each other too closely. The threshold should not be set too high 
    as we still wish to see refractory period violations due to poor spike sorting.
    Args:
        df (pandas DataFrame)
        date_key (str): columns key for dates in dataframe
        timestamp_key (str): column key for timestamps in dataframe
        neuron_key (str): neuron key for neuron labels in dataframe
        threshold (float): in s
        sampling_rate (float): in Hz
    Output:
        df (pandas Dataframe): dropped rows
    '''
    threshold = threshold*sampling_rate
    mask = df.groupby([neuron_key, date_key])['timestamps'].diff().fillna(1e6) > threshold
    df = df.loc[mask, :]
    return df

