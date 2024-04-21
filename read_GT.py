import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from post_processing import _detrend

def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


import numpy as np
import matplotlib.pyplot as plt

def read_ground_truth(path):
    with open(path, 'r') as file:
        truth_data = file.read().strip().split()
        
    data_list = [float(item) for item in truth_data]
    data_array = np.array(data_list)

    data_array = _detrend(data_array, 100)
    return data_array


