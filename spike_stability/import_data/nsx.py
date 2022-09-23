import os
import matplotlib.pyplot as plt
import numpy as np

from scipy import signal
from datetime import datetime
from spike_stability.util import brpylib

def read_nsx_datafile(datafile, elec_ids_='all'):
    brpylib_ver_req = "1.3.1"
    if brpylib.brpylib_ver.split('.') < brpylib_ver_req.split('.'):
        raise Exception("requires brpylib " + brpylib_ver_req + " or higher, please use latest version")                                      
    nsx_file = brpylib.NsxFile(datafile)
    # Extract data - note: data will be returned based on *SORTED* elec_ids, see cont_data['elec_ids']
    cont_data = nsx_file.getdata(elec_ids_)
    # unit = nev_file.extended_headers
    # Close the nsx file now that all data is out
    nsx_file.close()
    return cont_data
    
def give_correct_files_ns(files,
                       file_folder_name,
                       file_format='.nev',
                       nb_recordings=1):
    '''
    Function that returns all files within the file folder 
    that have the correct format.
    Args:
        files (list of str)
        file_format (str)
        file_folder_name (str)
        nb_recordings (int): the number of recordings in one day you wish to include
    Returns:
       correct_files (list of str): names of files to include in analysis
    '''
    correct_files = []
    counter = 0
    for file_name in sorted(files):
        if counter >= nb_recordings:
            break
        if file_name.endswith(file_format):
            if not '_sti' in file_name:
                correct_files.append(file_name)
                counter += 1
    if counter != nb_recordings:
        print(file_folder_name+'does not have an adequate number of recordings, returning empty list')
        return []
    else:
        return correct_files

def is_file_folder_before_date(file_folder_name,
                            cutoff_date=(2020, 4, 1)):
    '''
    Function to test whether the date corresponding to the files
    is before april 2020 or not. This is due to different channel
    indexing before and after april 2020.
    Args:
        file_folder_name (str)
    Returns:
        bool
    '''
    date_ = datetime(year=int('20'+file_folder_name[4:6]),
                        month=int(file_folder_name[:2]),
                        day=int(file_folder_name[2:4]))
    cutoff_date_ = datetime(year=cutoff_date[0],
                            month=cutoff_date[1],
                            day=cutoff_date[2])
    return date_ < cutoff_date_

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def fetch_nsx_data(data_basepath,
                   sampling_rate,
                   lowcut_filter,
                   highcut_filter,
                   order_filter,
                   channel,
                   channel_before_april,
                   file_format,
                   nb_recordings,
                   cutoff_date,
                   plot=True):
    '''
    Automate fetching data from .ns6 files.
    File folder should have individual folders for each 
    recording day. File names for these recordings should be 
    date in %m%d%y e.g. 052220. 
    Args:
        data_basepath (str): basepath to folder containing indivi folders
        sampling_rate (int)
        lowcut_filter (int): lower range for bandpass filtering of continuous signal e.g 4 if
        filtering for theta oscillations in the 4-8 Hz range
        highcut_filter (int): higher range 
        order_filter (int)
        channel (int): channel you wish to extract. For memory reasons we do one at a time
        channel_before_april (int): misalignment of channel numbers across days
        file_format (str)
        nb_recordings (int): number of recordings to include per day if multiple recordings
        cutoff_date (tuple): (Y, m, d) e.g. (2020, 04, 01)
        plot (bool): if wish to plot small part of continuous signal and filtered signal
    Returns:
        recording_stages (dict): dictionary with keys as dates and values are corresponding
        filtered signals
        channels_stages (dict): dictionary with keys as dates and values being chosen channel
    '''
    #channels 
    channels_here = [channel, channel_before_april]
    #init dics
    recording_stages = {}
    channels_stages = {}

    for dirpath, dirnames, files in os.walk(data_basepath):
        file_folder_name = dirpath.split('/')[-1][:6]
        correct_files = give_correct_files_ns(files,
                                           file_folder_name,
                                           file_format=file_format,
                                           nb_recordings=nb_recordings)
        for epoch_, file_name in enumerate(correct_files):
            file_folder_name_str = datetime.strptime(file_folder_name, '%m%d%y').strftime('%Y-%m-%d')
            #get file path 
            file_path = dirpath + '/' + file_name
            print(file_folder_name_str)
            #extract data
            cont_data_dict = read_nsx_datafile(file_path, elec_ids_=channels_here)
            #get channel names
            if is_file_folder_before_date(file_folder_name, cutoff_date=cutoff_date):
                #correct_channels = rename_channels(cont_data_dict['elec_ids'], channel_correspondance) 
                correct_channels = [channel_before_april]
            else:
                correct_channels = [channel]        
            #channels dic
            channels_stages[file_folder_name_str] = channels_stages.setdefault(file_folder_name_str, {})
            channels_stages[file_folder_name_str][epoch_] = []

            #list of arrays with 32 * 3600403 shape        
            recording_stages[file_folder_name_str] = recording_stages.setdefault(file_folder_name_str, {})
            recording_stages[file_folder_name_str][epoch_] = []

            for i, channel in enumerate(correct_channels): 
                print('channel '+str(channel))
                raw_ = cont_data_dict['data'][channels_here.index(channel), :]
                #data for lfp
                theta_ = butter_bandpass_filter(raw_, lowcut_filter, highcut_filter, sampling_rate, order=order_filter)
                if plot:
                    plt.plot(raw_[:30000])
                    plt.plot(theta_[:30000])
                    plt.show()
                del raw_ #for memory purposes, before appending current data
                #dics for lfps
                recording_stages[file_folder_name_str][epoch_].append(theta_)
                channels_stages[file_folder_name_str][epoch_].append(channel)

            #make into arrays
            recording_stages[file_folder_name_str][epoch_] = np.array(recording_stages[file_folder_name_str][epoch_])
    return recording_stages, channels_stages


#TODO fetching continuous nsx data on all relevant channels and performing spike extraction here in python