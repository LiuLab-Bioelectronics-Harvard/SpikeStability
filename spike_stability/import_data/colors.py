import json
import re

def color_dict(path_color_dics,
              reg_expression_naming='channel(\d+?)mouse(\d+?)',
              channel=None, mouse=None):
    '''
    Function to read color dictionaries for each channel of the mice.
    The text file should have the following structure:
    'channelXmouseY' = {categorie: {clusternb: color_hex, cluster_nb:hex}, categorie:{}, ..}
    In this particular usage categories include different clustering techniques to identify
    single units such as waveclus, wavemap calculated across days or for single days. We 
    wish to color code these clusters differently.
    Args:
        path_colors_dics (str): path to color dictionary
        reg_expression_naming (str): regular expression to find substring of key information in line of txt file
        channel (int): channel number if wish to get color dics associated with a specific channel
        mouse (int): mouse number if wish to get color dics associated with a specific mouse
    Returns:
        Corresponding dictionnary or list of dictionnaries
    '''
    with open(path_color_dics) as f:
        lines = f.readlines()
        final_color_dic = {}
        for line in lines:
            line = line.split('=')
            if mouse:
                if 'mouse'+str(mouse) in line[0]:
                    if 'channel'+str(channel) in line[0]:
                        return json.loads(f'''{line[1]}''')
                    else:
                        final_color_dic[line[0][1:-1]] = json.loads(f'''{line[1]}''')
            else:
                final_color_dic[re.search(reg_expression_naming,
                                          line[0]).group()] = json.loads(f'''{line[1]}''')
    return final_color_dic