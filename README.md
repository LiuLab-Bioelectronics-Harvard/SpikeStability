# spike_stability

## Overview
This repository contains a python package written along with the manuscript from Zhao, Tang, Tian, Partarrieu et al. ["Tracking neural activity from the same cells during the entire adult life of mice"](https://www.biorxiv.org/content/10.1101/2021.10.29.466524v1). It was developped in order to make a standalone package being able to do the various steps of the neural signal analysis. The ultimate goal is to make consistency/stability analysis of long-term neural recordings easier and more accessible. As this is adapted to a specific type of BMI technology, there are necessarily functions and parts of the package that might not generalize so well to your data.

## Installation
Before any of the following, make sure you have a working installation of python. If you're starting from scratch, the easiest will probably be to download the [Anaconda](https://www.anaconda.com/products/individual) toolkit which will also download what you need to be able to run the notebooks and the anaconda command prompt which may be useful in running some of the commands below.

### Option 1
For use of the latest version deployed here on github, I highly recommend using a python virtual environment in which case you may run (make sure you've changed the .yml file appropriately):
```
git clone https://github.com/LiuLab-Bioelectronics-Harvard/SpikeStability
cd SpikeStability
conda env create -f environment.yml
```
If you want to be able to use this environment with the jupyter-notebooks you'll need to run the command (once you've activated the environment)
```
python -m ipykernel install --user --name=your_env_name_here
```
### Option 2
You can always [create your own environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#saving-environment-variables), activate it and then run
```
pip install -r requirements.txt
```

## Data
This repository relies on various types of data to use all its functionalities. Due to the size of the data files associated with this work, it is inconvenient to host these on github. Example files are provided in the data/ repository but people should contact the corresponding author of the manuscript to inquire about access to the large volume of data analyzed in this work.

Follow the tutorials and example worflows in the jupyter notebooks for more details about importing various data types.

## Package structure 
This package is divided into several main submodules, each performing some clear task: import, cluster, preprocessing and quality. 

### import_data
Contains necessary import functions to process .mat or .nsX data provided they have a certain structure. This is for importing data from *files* in separate directories, and has been written with our particular file structure in mind. You can always pass custom functions to read your own .mat file strucure if need be and change regular expressions written to fetch the date from the file path or channel information from the file name.

More details are provided in the example workflows.

### preprocessing
Contains a few preprocessing functions for basic quality control of the recordings: removing artifacts from the signal, ensuring spike alignment, ...

### cluster
This has various clustering methods used to differentiate neurons (spike sorting) given the extracted waveforms, timestamps and dates of your neural recordings. The main algorithm featured here is wavemap (basically UMAP and leiden clustering on spike waveforms). Users may pass their own cluster labels obtained with their favorite spike sorting algorithm to the functions in the stability sub-module.

### stability 
The heart of this repository. This contains multiple methods allowing the user to perform consistency analysis given spikes, dates, timestamps and cluster labels. 

The consistency analysis is composed of multiple different scripts and functions. These contain:
- Dimensionality reduction over recording days
- Features computed from spike waveforms analysed over time
- ISI distribution profiles analyzed over time
- Waveform auto and cross-correlation analysis
- Cluster quality metrics such as L-ratio, silhouette score and others computed over time

### util
Miscellaneous utility functions.

## Other information

### Citation
If SpikeStability has been useful for your work:

> Siyuan Zhao, Xin Tang, Weiwen Tian, Sebastian Partarrieu, Shiqi Guo, Ren Liu, Jaeyong Lee, Zuwan Lin, Jia Liu
bioRxiv 2021.10.29.466524; doi: https://doi.org/10.1101/2021.10.29.466524

### Contact
Feel free to open an issue on the github and we'll respond as quickly as possible.
