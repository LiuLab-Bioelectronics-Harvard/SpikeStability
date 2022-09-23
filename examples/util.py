from scipy.stats import pearsonr
from ismember import ismember
import pylab
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from probeinterface import generate_multi_columns_probe
import random

def waveform_similarity_cal(choosen_day, template_day0, template_day_follow):
    
    r_values = []
    p_values = []
    for idx, choosen_day_id in enumerate(choosen_day):
        template_x = np.reshape(template_day0[choosen_day_id[0]].T,(-1,))
        template_y = np.reshape(template_day_follow[choosen_day_id[1]].T,(-1,))
        
        r_value, p_value = pearsonr(template_x, template_y)
        
        r_values.append(r_value)
        p_values.append(p_value)

    return np.array(r_values), np.array(p_values)

def waveform_overlay_plot(stable_info, stable_neuron_ids, y_scale_factor=0.8, x_scale_factor=1,
                  ylim=[-100,200], y_displace=5, alpha_lim=[0.5,1],save_path=None):
    array_channels = [[5,3,1,7,9,11],[17,15,13,19,21,23],[29,27,25,28,26,24],[18,20,22,16,14,12],[6,8,10,4,2,0]]
    sensor_location = [[0,250],[0,125],[0,0],[50,250],[50,125],[50,0]] # not micrometer, just use it as fraction to plot
    cm = pylab.get_cmap('gist_rainbow')
    NUM_COLORS = len(stable_neuron_ids)
    colors = []
    for i in range(NUM_COLORS):
        colors.append(cm(1. * i / NUM_COLORS))
        
    alpha_s = np.arange(alpha_lim[0],alpha_lim[1]+(alpha_lim[1]-alpha_lim[0])/(len(stable_info)-1),
                        (alpha_lim[1]-alpha_lim[0])/(len(stable_info)-1))
    
    fig, axes = plt.subplots(len(stable_neuron_ids),1,
                             figsize=(6,10*len(stable_neuron_ids)))
    
    save_name = 'waveform_similarity_overlay_'
    for align_id in range(len(stable_info)):
        info_day = stable_info[align_id]
        day_name = info_day['month_name'].values[0]
        save_name = save_name + '_' + day_name
    
    for neuron_id_id, neuron_id in enumerate(stable_neuron_ids):
        for align_id in range(len(stable_info)):
            info_month = stable_info[align_id]
            day_name = info_month['month_name'].values[0]
            ax = axes[neuron_id_id]
            array_id = info_month.loc[info_month['neuron_id']==neuron_id]['array_id'].values[0]
            plot_channel_ids = array_channels[array_id]
            template = info_month.loc[info_month['neuron_id']==neuron_id]['template'].values[0][:, plot_channel_ids]
            tps = np.arange(template.shape[0])
            
            for idx in range(6):
                x = (tps-template.shape[0]/3)*x_scale_factor+sensor_location[idx][0]
                y = y_scale_factor*(template[:,idx]+sensor_location[idx][1] - y_displace*align_id)
                ax.plot(x, y,linewidth=3,c=colors[neuron_id_id], alpha=alpha_s[align_id])
                
        ax.set_title(f'neuron {neuron_id_id+1}, array {array_id+1}')
        ax.set_ylim(ylim)
        ax.axis('equal')
        ax.axis('off')
        
    if(save_path is not None):
        plt.savefig(save_path)
        
        
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
        

def location_cal(info, degree=1):
    sensor_positions = np.array([[0,60],[0,30],[0,0],[30,60],[30,30],[30,0]])
    
    array_channels = [[5,3,1,7,9,11],[17,15,13,19,21,23],[29,27,25,28,26,24],[18,20,22,16,14,12],[6,8,10,4,2,0]]
    
    location_day = []
    
    for neuron_id in range(len(info)):
        info_unit = info.iloc[neuron_id]
        array_id = info_unit['array_id']
        plot_channel_ids = array_channels[array_id]
        template = info_unit['template'][:, plot_channel_ids]
        NumChannels = template.shape[1]
        amplitudes = np.max(template,axis=0) - np.min(template,axis=0)
        x = np.sum(np.array([sensor_positions[i,0]*(amplitudes[i]**degree) for i in range(NumChannels)]))
        x /= np.sum(np.array([(amplitudes[i]**degree) for i in range(NumChannels)]))
        y = np.sum(np.array([sensor_positions[i,1]*(amplitudes[i]**degree) for i in range(NumChannels)]))
        y /= np.sum(np.array([(amplitudes[i]**degree) for i in range(NumChannels)]))

        location_day.append([x,y])
        
    return np.array(location_day)

def plot_probe(probe, ax=None, contacts_colors=None,
               with_channel_index=False, with_contact_id=False,
               with_device_index=False,
               first_index='auto',
               contacts_values=None, cmap='viridis',
               title=True, contacts_kargs={}, probe_shape_kwargs={},
               xlims=None, ylims=None, zlims=None,
               show_channel_on_click=False):

    import matplotlib.pyplot as plt
    if probe.ndim == 2:
        from matplotlib.collections import PolyCollection
    elif probe.ndim == 3:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        if probe.ndim == 2:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
        else:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        fig = ax.get_figure()

    if first_index == 'auto':
        if 'first_index' in probe.annotations:
            first_index = probe.annotations['first_index']
        elif probe.annotations.get('manufacturer', None) == 'neuronexus':
            # neuronexus is one based indexing
            first_index = 1
        else:
            first_index = 0
    assert first_index in (0, 1)

    _probe_shape_kwargs = dict(
        facecolor='white', edgecolor='k', lw=0.5, alpha=1)
    _probe_shape_kwargs.update(probe_shape_kwargs)

    _contacts_kargs = dict(alpha=1, edgecolor=[1, 1, 1], lw=0.5)
    _contacts_kargs.update(contacts_kargs)

    n = probe.get_contact_count()

    if contacts_colors is None and contacts_values is None:
        contacts_colors = ['lightgray'] * n
    elif contacts_colors is not None:
        contacts_colors = contacts_colors
    elif contacts_values is not None:
        contacts_colors = None

    # probe shape
    planar_contour = probe.probe_planar_contour
    if planar_contour is not None:
        if probe.ndim == 2:
            poly_contour = PolyCollection(
                [planar_contour], **_probe_shape_kwargs)
            ax.add_collection(poly_contour)
        elif probe.ndim == 3:
            poly_contour = Poly3DCollection(
                [planar_contour], **_probe_shape_kwargs)
            ax.add_collection3d(poly_contour)
    else:
        poly_contour = None
        
    vertices = probe.get_contact_vertices()
    if probe.ndim == 2:
        poly = PolyCollection(
            vertices, color=contacts_colors, **_contacts_kargs)
        ax.add_collection(poly)
    elif probe.ndim == 3:
        poly = Poly3DCollection(
            vertices, color=contacts_colors, **_contacts_kargs)
        ax.add_collection3d(poly)

    if contacts_values is not None:
        poly.set_array(contacts_values)
        poly.set_cmap(cmap)

    if show_channel_on_click:
        assert probe.ndim == 2, 'show_channel_on_click works only for ndim=2'
        def on_press(event): return _on_press(probe, event)
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)

    if with_channel_index or with_contact_id or with_device_index:
        if probe.ndim == 3:
            raise NotImplementedError('Channel index is 2d only')
        for i in range(n):
            txt = []
            if with_channel_index:
                txt.append(f'{i + first_index}')
            if with_contact_id and probe.contact_ids is not None:
                contact_id = probe.contact_ids[i]
                txt.append(f'id{contact_id}')
            if with_device_index and probe.device_channel_indices is not None:
                chan_ind = probe.device_channel_indices[i]
                txt.append(f'{chan_ind}')
            txt = '\n'.join(txt)
            x, y = probe.contact_positions[i]
            ax.text(x, y, txt, ha='center', va='center', c='w',clip_on=True)

    if xlims is None or ylims is None or (zlims is None and probe.ndim == 3):
        xlims, ylims, zlims = get_auto_lims(probe)

    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)

    if title:
        ax.set_title(probe.get_title())

    return ax, poly, poly_contour

def get_auto_lims(probe, margin=20):
    positions = probe.contact_positions
    planar_contour = probe.probe_planar_contour

    xlims = np.min(positions[:, 0]), np.max(positions[:, 0])
    ylims = np.min(positions[:, 1]), np.max(positions[:, 1])
    zlims = None

    if probe.ndim == 3:
        zlims = np.min(positions[:, 2]), np.max(positions[:, 2])

    if planar_contour is not None:

        xlims2 = np.min(planar_contour[:, 0]), np.max(planar_contour[:, 0])
        xlims = min(xlims[0], xlims2[0]), max(xlims[1], xlims2[1])

        ylims2 = np.min(planar_contour[:, 1]), np.max(planar_contour[:, 1])
        ylims = min(ylims[0], ylims2[0]), max(ylims[1], ylims2[1])

        if probe.ndim == 3:
            zlims2 = np.min(planar_contour[:, 2]), np.max(planar_contour[:, 2])
            zlims = min(zlims[0], zlims2[0]), max(zlims[1], zlims2[1])

    xlims = xlims[0] - margin, xlims[1] + margin
    ylims = ylims[0] - margin, ylims[1] + margin

    if probe.ndim == 3:
        zlims = zlims[0] - margin, zlims[1] + margin

        # to keep equal aspect in 3d
        # all axes have the same limits
        lims = min(xlims[0], ylims[0], zlims[0]), max(
            xlims[1], ylims[1], zlims[1])
        xlims, ylims, zlims = lims, lims, lims

    return xlims, ylims, zlims


def unit_position_plot(location_day0, location_day1, array_ids, day0_name, day_follow_name, with_device_index=True, 
                       colors = ['darkblue','red'], s=[100,80],linewidth=1, save_path=None):
    device_channel_indices = [[1,3,5,11,9,7],[13,15,17,23,21,19],[25,27,29,24,26,28],
                              [22,20,18,12,14,16],[10,8,6,0,2,4]]
    ArrayNum = 5
    fig,axs = plt.subplots(1,ArrayNum,figsize=(ArrayNum*2,3))

    for array_id in range(ArrayNum):
        mesh_probe = generate_multi_columns_probe(num_columns=2,
                                          num_contact_per_column=3,
                                          xpitch=30, 
                                          ypitch=30,
                                          contact_shapes='circle', contact_shape_params={'radius': 7.5})
        mesh_probe.set_device_channel_indices([device_channel_indices[array_id][i]+1 
                                               for i in range(len(device_channel_indices[array_id]))])
        mesh_probe.probe_planar_contour = None
        
        ax = axs[array_id]
        plot_probe(mesh_probe, with_device_index=with_device_index,ax=ax)
        ax.set_title(f'Array{array_id+1}')
        array_location_day0 = location_day0[array_ids==array_id]
        array_location_day1 = location_day1[array_ids==array_id]

        for neuron_id in range(array_location_day0.shape[0]):
            label = f'neuron{neuron_id+1}'
            ax.scatter(array_location_day0[neuron_id,0], array_location_day0[neuron_id,1], 
                           marker='.',s=s[0], color=colors[0], label=label)
        for neuron_id in range(array_location_day0.shape[0]):
            label = f'neuron{neuron_id+1}'
            ax.scatter(array_location_day1[neuron_id,0], array_location_day1[neuron_id,1], 
                           marker='.',s=s[1], color=colors[1], label=label)
            
            ax.plot([array_location_day0[neuron_id,0], array_location_day1[neuron_id,0]], 
                    [array_location_day0[neuron_id,1], array_location_day1[neuron_id,1]], c=colors[1],
                   linewidth=linewidth)
            
        ax.axis('off')
         
    fig.suptitle(f'{day0_name} - {day_follow_name}')
    plt.tight_layout()
    if(save_path is not None):
        plt.savefig(save_path)
        
        
def visual_matching_2D_plot_all_mice(neuron_pair_day0, neuron_pair_day_follow, day0_name, day_follow_name, 
                            shank_ids, color='#1A6FDF',save_path=None):
    fig, ax = plt.subplots(figsize=(5,5))
    neuron_num = len(neuron_pair_day0)
    ax.scatter(neuron_pair_day0, 
               neuron_pair_day_follow, marker='|', c=color, s=40, linewidth=1)

    ax.set_xlim(-0.5,neuron_num-0.5)
    ax.set_ylim(-0.5,neuron_num-0.5)
    ax.set_xlabel(f'response neuron index ({day0_name})')
    ax.set_ylabel(f'index of best matcch ({day_follow_name})')
    ax.set_title(f'')
    if(save_path is not None):
        plt.savefig(save_path)
        
def feature_plot_multidays(features, plot_mouse_ids, feature_name, days, color='#E51A1C', save_path=None):
    if(feature_name == 'peak_to_valley'):
        features = features*1000
    elif(feature_name == 'repolarization_slope' or feature_name == 'recovery_slope'):
        features = features*1e-6
    elif(feature_name== 'peak_trough_ratio'):
        features = np.abs(features)
    num_days = features.shape[0]
    
    fig, ax = plt.subplots(figsize=(8,2))
    features_concat = []
    for mouse_id in np.unique(plot_mouse_ids):
        mouse_indices = plot_mouse_ids==mouse_id
        mouse_features = features[:,mouse_indices]
        nan_indices = np.isnan(mouse_features)
        day_ids = np.where(np.sum(nan_indices,axis=1) < mouse_features.shape[1])[0]
        select_feature = mouse_features[day_ids,:]
        select_indices = np.where(~np.isnan(np.sum(select_feature,axis=0)))[0]
        features_ = select_feature[:,select_indices]
        feature_add_day_info = np.empty((num_days, len(select_indices)))*np.nan
        feature_add_day_info[day_ids,:] = features_
        features_concat.append(feature_add_day_info)
        ax.plot(day_ids,features_, linewidth=1, alpha=0.1, c=color,marker='.')
        
    features_concat = np.hstack(features_concat)
    feature_mean = np.nanmean(features_concat,axis=1)
    feature_std = np.nanstd(features_concat,axis=1)
    ax.errorbar(range(len(feature_mean)),feature_mean, yerr=feature_std,
         c=color, marker='.',capsize=3, linewidth=2, markersize=10)
    ax.set_xlabel('Month')
    ax.set_ylabel(feature_name)
    ax.set_xticks(range(num_days), days)
    ax.set_xlim([-0.5, num_days-0.5])
    
    ylim = ax.get_ylim()
    ax.set_ylim([ylim[0]-0.5*(ylim[1]-ylim[0]), ylim[1]+0.5*(ylim[1]-ylim[0])])
    
    if(save_path is not None):
        plt.savefig(save_path)

    
    