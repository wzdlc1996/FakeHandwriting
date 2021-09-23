import fakeChar as hw
from PIL import Image
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def rgb2gray(rgb):
    """
    Convert the RGB array into normalized gray array

    :param rgb: the rgb array of size (weight, height, 3)
    :return: gray array of size (weight, height). the values are normalized in [0, 1]
    """
    rgb = np.array(rgb)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    gamma = 2.2
    fac = 1 + 1.5 ** gamma + 0.6 ** gamma
    gray = np.power((r ** gamma + (1.5 * g) ** gamma + (0.6 * b) ** gamma) / fac, 1 / gamma)
    return gray / np.max(gray)


def neighborLeadFlip(arr, thres=0.6):
    """
    Hold the arr[i] as 0 if the share of 0 in its neighbor is greater than thres, otherwise 1
    """
    w_s = 3
    pd = (w_s - 1) // 2
    arr_padded = np.pad(arr, [(pd, pd), (pd, pd)], constant_values=1)
    nei = sliding_window_view(arr_padded, (w_s, w_s))
    pval = 1 - (np.sum(nei, axis=(2, 3)) - arr) / (w_s ** 2 - 1)
    return 1 - np.int_(pval > thres)




# make a generator without random
gener = hw.char2imgFromFont(widCent=50, widRang=0, height=50, font_size=35)
w, h = gener.genSize()

char, size = gener("啊")
nch = Image.new("RGB", color="white", size=size)
nch.paste(char, char)
# nch = nch.convert("1")
nch = rgb2gray(nch)
nch = np.int_(nch > 0.95)



import matplotlib.pyplot as plt

plt.matshow(np.int_(nch))
plt.show()

plt.matshow(neighborLeadFlip(nch))
plt.show()
