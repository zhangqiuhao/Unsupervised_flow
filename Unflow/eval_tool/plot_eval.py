import csv
import os
import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from matplotlib.path import Path
import matplotlib.patches as patches


def read_csv(file_path):
    file_content = []
    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            file_content.append([float(row['Step']), float(row['Value'])])
    return np.asarray(file_content)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, np.full(N,x[0])))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


dst = '/home/zqhyyl/evaluation/eval_csv/'
networks = ['FN_4L_geo', 'FN_4L_gm']

files = dict()
for network in networks:
    network_dir = dst + network
    for file in os.listdir(network_dir):
        if file.split('-')[-1].split('_')[0] != 'weight':
            value = read_csv(network_dir + '/' + file)
            weight_file = file.split('loss')[0] + 'weight' + file.split('loss')[-1]
            if os.path.exists(network_dir + '/' + weight_file):
                weight = read_csv(network_dir + '/' + weight_file)
                #value[:, 1] = value[:, 1] / weight[:, 1]
                print('Divide Weights')
            network_name = network + file.split('-')[-1].split('loss')[1].split('_1')[0]
            files[network_name] = value

outplot_value = 'combined'
plt.figure(figsize=(10, 2))
plt.title(outplot_value + '_loss')
plt.xlabel('Epochs', horizontalalignment='right', x=1.0)
plt.ylabel('Value')

min_timestamp = 100000
count = 0
files_plot = []
for name, dict_ in files.items():
    if name.split('_')[-1] == outplot_value:
        files_plot.append(name)
        if max(dict_[:,0]) < min_timestamp:
            min_timestamp = max(dict_[:,0])
        count += 1

#print(files['FN_4L_geo_combined'])
files['FN_4L_geo_combined'][:,1] = files['FN_4L_geo_combined'][:,1] - files['FN_4L_geo_photo'][:,1]

for name in files_plot:
    print(name)
    dict_ = files[name]
    plot = []
    dict_[:,1] = running_mean(dict_[:,1], 10)
    for x in range(1, len(dict_)-1):
        if dict_[x][0] < min_timestamp and x % 5 == 0:
            plot.append([dict_[x][0], dict_[x][1]])
    plot = np.asarray(plot)
    plt.plot(plot[:,0] * 4 /18932, plot[:,1], label=name)

plt.legend()
plt.savefig('/home/zqhyyl/plot.svg',bbox_inches='tight')
