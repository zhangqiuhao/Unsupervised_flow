import numpy as np
import os

directory = '/home/zhang/pcl_data/02/data/'

for filename in os.listdir(directory):
    print filename
    os.rename(directory + filename, directory + '0000' + filename)
