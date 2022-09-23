import seaborn as sns

def hex_to_RGB(hex):
    ''' "#FFFFFF" -> [255,255,255] '''
    # Pass 16 to the integer function for change of base
    return [int(hex[i:i+2], 16) for i in range(1,6,2)]


def RGB_to_hex(RGB):
    ''' [255,255,255] -> "#FFFFFF" '''
    # Components need to be integers for hex to make sense
    RGB = [int(x) for x in RGB]
    return "#"+"".join(["0{0:x}".format(v) if v < 16 else
                        "{0:x}".format(v) for v in RGB])

def color_dict(gradient):
    ''' Takes in a list of RGB sub-lists and returns dictionary of
      colors in RGB and hex form for use in a graphing function
      defined later on '''
    return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
            "r":[RGB[0] for RGB in gradient],
            "g":[RGB[1] for RGB in gradient],
            "b":[RGB[2] for RGB in gradient]}

def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    ''' returns a gradient list of (n) colors between
      two hex colors. start_hex and finish_hex
      should be the full six-digit color string,
      inlcuding the number sign ("#FFFFFF") '''
    # Starting and ending colors in RGB form
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
    # Initilize a list of the output colors with the starting color
    RGB_list = [s]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    for t in range(1, n):
        # Interpolate RGB vector for color at the current value of t
        curr_vector = [
            int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
            for j in range(3)
        ]
        # Add it to our list of output colors
        RGB_list.append(curr_vector)
    return color_dict(RGB_list)

def color_dict_list(color_dicts, select_mouse, colors_name):
    '''
    Reformat dictionary of color mapping dictionaries to list.
    Dict format should be one impored by import_data.colors color_dict function
    Args:
        color_dics (dict of dicts)
        select_mouse (str): select elements of color_dics based on this substring
        colors_name (str): once appropriate channel/mouse has been selected, identifier for method
    Returns:
        cluster_colors_all (dict of dict): per channel. Individual dicts have cluster identifiers (int) as keys
        and values are colors in hex
     '''
    cluster_colors_all = {}
    for key, val in color_dicts.items():
        if select_mouse in key:
            print(f'Adding color dic associated to {key} & {colors_name} to dict')
            current_dic = dict([(int(key_), val_) for key_, val_ in val[colors_name].items()])
            cluster_colors_all[key.split('channel')[1].split('mouse')[0]] = current_dic
    return cluster_colors_all

def create_neuron_color_dict(palette, neuron_values):
    '''
    Function to create mapping dict between neuron value and colors of a 
    palette.
    Args: 
        palette (str): to be passed to sns.color_palette e.g. "husl"
        neuron_values (list): unique neuron label values to map colors to
    '''
    color_palette = sns.color_palette(palette, len(neuron_values))
    color_dict = dict([(neuron_val, color) for neuron_val, color in zip(neuron_values, color_palette)])
    return color_dict