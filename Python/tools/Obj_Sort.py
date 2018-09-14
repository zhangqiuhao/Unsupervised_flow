from os import listdir
from os.path import isfile, join

with open('/home/klein/Desktop/train_mapping.txt', 'r') as f:
    mapping = [line.strip() for line in f]


with open('/home/klein/Desktop/train_rand.txt', 'r') as f:
    rand = [line.strip().split(',') for line in f]
rand = rand[0]

gm_dir = '/home/klein/U/object/gridmaps_cropped/'
gm_files = [int(f[:-4]) for f in listdir(gm_dir) if isfile(join(gm_dir, f))]


with open('/home/klein/U/object/all_available_data_cars.txt', 'w') as cars:
    for i, map in enumerate(mapping):

        for j, rnd in enumerate(rand):
            if int(rnd) == i+1:
                index = j
                break

        if index in gm_files:
            cars.write('/home/klein/U/object/gridmaps_cropped/'+str(index).zfill(6) + '.png,/home/klein/U/object/labels_img/'+str(index).zfill(6) + '.png\n')
            print(i, index)



