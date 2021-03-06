"""
Functions based on
https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/mean_waveforms/waveform_metrics.py
22/04/2020
"""

import numpy as np
from scipy.stats import linregress
import pandas as pd

all_1D_features = ['peak_to_valley', 'halfwidth', 'peak_trough_ratio',
                   'repolarization_slope', 'recovery_slope']


def features(df, spike_cols, sampling_frequency):
    '''
    Calculate features for all waveforms and add to dataframe
    Parameters
    ----------
    df : pandas DataFrame
    spike_cols: list of str
    sampling_frequency: float

    Returns
    -------
    df: pandas DataFrame with feature columns
    '''
    X_features = features_5(df[spike_cols].values, sampling_frequency)
    X_features = pd.DataFrame(X_features)
    X_features['amplitude'] = np.abs(df.loc[:, spike_cols].max(axis=1) - df.loc[:, spike_cols].min(axis=1))
    df[X_features.columns] = X_features
    print('Information about NaN values in feature calculation due to unorthodox waveforms: ')
    print(X_features.isna().sum())
    return df


def features_5(waveforms, sampling_frequency, feature_names=None,
                       recovery_slope_window=0.7):
    """ Calculate features for all waveforms
    Parameters
    ----------
    waveforms  : numpy.ndarray (num_waveforms x num_samples)
        waveforms to compute features for
    sampling_frequency  : float
        rate at which the waveforms are sampled (Hz)
    feature_names : list or None (if None, compute all)
        features to compute
    recovery_slope_window : float
        windowlength in ms after peak wherein recovery slope is computed
    Returns
    -------
    metrics : dict  (num_waveforms x num_metrics)
        Dictionary with computed metrics. Keys are the metric names, values are the computed features
    """
    metrics = dict()

    if feature_names is None:
        feature_names = all_1D_features
    else:
        for name in feature_names:
            assert name in all_1D_features, f'{name} not in {all_1D_features}'

    if 'peak_to_valley' in feature_names:
        metrics['peak_to_valley'] = peak_to_valley(waveforms=waveforms,
                                                   sampling_frequency=sampling_frequency)
    if 'peak_trough_ratio' in feature_names:
        metrics['peak_trough_ratio'] = peak_trough_ratio(waveforms=waveforms)

    if 'halfwidth' in feature_names:
        metrics['halfwidth'] = halfwidth(waveforms=waveforms,
                                         sampling_frequency=sampling_frequency)

    if 'repolarization_slope' in feature_names:
        metrics['repolarization_slope'] = repolarization_slope(
            waveforms=waveforms,
            sampling_frequency=sampling_frequency,
        )

    if 'recovery_slope' in feature_names:
        metrics['recovery_slope'] = recovery_slope(
            waveforms=waveforms,
            sampling_frequency=sampling_frequency,
            window=recovery_slope_window,
        )
    return metrics


def peak_to_valley(waveforms, sampling_frequency):
    """ Time between trough and peak
    Parameters
    ----------
    waveforms  : numpy.ndarray (num_waveforms x num_samples)
        waveforms to compute feature for
    sampling_frequency  : float
        rate at which the waveforms are sampled (Hz)
    Returns
    -------
    np.ndarray (num_waveforms)
        peak_to_valley in seconds
    """
    trough_idx, peak_idx = _get_trough_and_peak_idx(waveforms)
    ptv = (peak_idx - trough_idx) * (1 / sampling_frequency)
    ptv[ptv == 0] = np.nan
    return ptv


def peak_trough_ratio(waveforms):
    """ Ratio peak heigth and trough depth
    Assumes baseline is 0
    Parameters
    ----------
    waveforms  : numpy.ndarray (num_waveforms x num_samples)
        waveforms to compute feature for
    Returns
    -------
    np.ndarray (num_waveforms)
        Peak to trough ratio
    """
    trough_idx, peak_idx = _get_trough_and_peak_idx(waveforms)
    ptratio = np.empty(trough_idx.shape[0])
    ptratio[:] = np.nan
    for i in range(waveforms.shape[0]):
        if peak_idx[i] == 0 and trough_idx[i] == 0:
            continue
        ptratio[i] = waveforms[i, peak_idx[i]] / waveforms[i, trough_idx[i]]

    return ptratio


def halfwidth(waveforms, sampling_frequency, return_idx=False):
    """
    Width of waveform at its half of amplitude
    Computes the width of the waveform peak at half it's height
    Parameters
    ----------
    waveforms  : numpy.ndarray (num_waveforms x num_samples)
        waveforms to compute features for
    sampling_frequency  : float
        rate at which the waveforms are sampled (Hz)
    return_idx : bool
        if true, also returns index of threshold crossing before and
        index of threshold crossing after peak
    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray, np.ndarray)
        Halfwidth of the waveforms or (Halfwidth of the waveforms,
        index_cross_pre_peak, index_cross_post_peak)
    """
    trough_idx, peak_idx = _get_trough_and_peak_idx(waveforms)
    hw = np.empty(waveforms.shape[0])
    hw[:] = np.nan
    cross_pre_pk = np.empty(waveforms.shape[0], dtype=int)
    cross_post_pk = np.empty(waveforms.shape[0], dtype=int)

    for i in range(waveforms.shape[0]):
        if peak_idx[i] == 0:
            cross_pre_pk[i] = 0
            cross_post_pk[i] = 0
            continue
        trough_val = waveforms[i, trough_idx[i]]
        threshold = 0.5 * trough_val  # threshold is half of peak heigth (assuming baseline is 0)

        cpre_idx = np.where(waveforms[i, :trough_idx[i]] < threshold)[0]
        cpost_idx = np.where(waveforms[i, trough_idx[i]:] < threshold)[0]

        if len(cpre_idx) == 0 or len(cpost_idx) == 0:
            continue

        cross_pre_pk[i] = cpre_idx[0] - 1  # last occurence of waveform lower than thr, before peak
        cross_post_pk[i] = cpost_idx[-1] + 1 + trough_idx[i]  # first occurence of waveform lower than peak, after peak

        hw[i] = (cross_post_pk[i] - cross_pre_pk[i]) * (1 / sampling_frequency)  # + peak_idx[i]

    if not return_idx:
        return hw
    else:
        return hw, cross_pre_pk, cross_post_pk


def repolarization_slope(waveforms, sampling_frequency, return_idx=False):
    """
    Return slope of repolarization period between trough and baseline
    After reaching it's maxumum polarization, the neuron potential will
    recover. The repolarization slope is defined as the dV/dT of the action potential
    between trough and baseline.
    Optionally the function returns also the indices per waveform where the
    potential crosses baseline.

    Parameters
    ----------
    waveforms  : numpy.ndarray (num_waveforms x num_samples)
        waveforms to compute features for
    sampling_frequency  : float
        rate at which the waveforms are sampled (Hz)
    return_idx : bool
        if true, also returns index of threshold crossing before and
        index of threshold crossing after peak
    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
        Repolarization slope of the waveforms or (Repolarization slope of the waveforms,
        return to base index)
    """
    trough_idx, peak_idx = _get_trough_and_peak_idx(waveforms)

    rslope = np.empty(waveforms.shape[0])
    rslope[:] = np.nan
    return_to_base_idx = np.empty(waveforms.shape[0], dtype=np.int)
    return_to_base_idx[:] = 0

    time = np.arange(0, waveforms.shape[1]) * (1 / sampling_frequency)  # in s
    for i in range(waveforms.shape[0]):
        if trough_idx[i] == 0:
            continue

        rtrn_idx = np.where(waveforms[i, trough_idx[i]:] >= 0)[0]
        if len(rtrn_idx) == 0:
            continue

        return_to_base_idx[i] = rtrn_idx[0] + trough_idx[i]  # first time after  trough, where waveform is at baseline

        if return_to_base_idx[i] - trough_idx[i] < 2:
            continue
        rslope[i] = linregress(time[trough_idx[i]:return_to_base_idx[i]],
                               waveforms[i, trough_idx[i]: return_to_base_idx[i]])[0]

    if not return_idx:
        return rslope
    else:
        return rslope, return_to_base_idx


def recovery_slope(waveforms, sampling_frequency, window):
    """
    Return the recovery slope of input waveforms. After repolarization,
    the neuron hyperpolarizes untill it peaks. The recovery slope is the
    slope of the actiopotential after the peak, returning to the baseline
    in dV/dT. The slope is computed within a user-defined window after
    the peak.
    Takes a numpy array of waveforms and returns an array with
    recovery slopes per waveform.
    Parameters
    ----------
    waveforms  : numpy.ndarray (num_waveforms x num_samples)
        waveforms to compute features for
    sampling_frequency  : float
        rate at which the waveforms are sampled (Hz)
    window : float
        length after peak wherein to compyte recovery slope (ms)
    Returns
    -------
    np.ndarray
        Recovery slope of the waveforms
    """
    _, peak_idx = _get_trough_and_peak_idx(waveforms)
    rslope = np.empty(waveforms.shape[0])
    rslope[:] = np.nan

    time = np.arange(0, waveforms.shape[1]) * (1 / sampling_frequency)  # in s

    for i in range(waveforms.shape[0]):
        if peak_idx[i] == 0:
            continue
        max_idx = int(peak_idx[i] + ((window / 1000) * sampling_frequency))
        max_idx = np.min([max_idx, waveforms.shape[1]])
        slope = _get_slope(time[peak_idx[i]:max_idx], waveforms[i, peak_idx[i]:max_idx])
        rslope[i] = slope[0]
    return rslope


def _get_slope(x, y):
    """
    Retrun the slope of x and y data, using scipy.signal.linregress
    """
    slope = linregress(x, y)
    return slope


def _get_trough_and_peak_idx(waveform):
    """
    Return the indices into the input waveforms of the detected troughs
    (minimum of waveform) and peaks (maximum of waveform, after trough).
    Assumes negative troughs and positive peaks
    Returns 0 if not detected
    """
    trough_idx = np.argmin(waveform, axis=1)
    peak_idx = -1 * np.ones(trough_idx.shape, dtype=int)  # int, these are used for indexing
    for i, tridx in enumerate(trough_idx):
        if tridx == waveform.shape[1] - 1:
            trough_idx[i] = 0
            peak_idx[i] = 0
            continue
        idx = np.argmax(waveform[i, tridx:])
        peak_idx[i] = idx + tridx
    return trough_idx, peak_idx