import numpy as np
from tools.NumpyHelpers import rotationMatrixToEulerAngles

np.set_printoptions(suppress=True)

############
in_file = '/home/klein/U/depthimageFlow/eval/'
x_thres = 1
angle_thres = 1.2
############

with open(in_file + 'all.csv', 'r') as f:
    in_data = [line for line in f]

data_sd = []
data_ld = []

for line in in_data:
    tf_string = line.split(',')[-1]

    tf = np.matrix((np.reshape([float(x) for x in tf_string.strip().split()], (4, 4))))

    x = tf[0, 3]
    _, _, angle = np.degrees(rotationMatrixToEulerAngles(tf[0:3,0:3]))

    if x > x_thres or angle > angle_thres:
        data_ld.append(line)
    else:
        data_sd.append(line)

print('Separated into:')
print(len(data_ld), 'large Displacement examples')
print(len(data_sd), 'small Displacement examples')

with open(in_file+'small_displacements.csv', 'w') as file:
    for item in data_sd:
        file.write(item)

with open(in_file+'large_displacements.csv', 'w') as file:
    for item in data_ld:
        file.write(item)