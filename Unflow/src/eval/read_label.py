import numpy as np


def Read_Label(data_list, data_num, img_size):
    data = data_list[data_num]
    objects = []
    (_, height, width, _) = img_size.shape
    res = 60.0 / float(height)

    if data is not None:
        for object_data in data:
            mid = np.array((width / 2.0 + object_data[5] / res, height - object_data[6] / res))

            yaw_angle = object_data[7]
            angle = - (np.pi / 2 + yaw_angle)

            c, s = np.cos(angle), np.sin(angle)
            R = np.array(((c, s), (-s, c)))

            w = object_data[3] / res / 2.0
            l = object_data[4] / res / 2.0

            p_ul = R.dot(np.array(((-w), (-l)))) + mid
            p_ur = R.dot(np.array(((+w), (-l)))) + mid
            p_dl = R.dot(np.array(((-w), (+l)))) + mid
            p_dr = R.dot(np.array(((+w), (+l)))) + mid

            p_ul = [int(i) for i in p_ul]
            p_ur = [int(i) for i in p_ur]
            p_dl = [int(i) for i in p_dl]
            p_dr = [int(i) for i in p_dr]

            objects.append([object_data[0], object_data[1], p_ul, p_ur, p_dl, p_dr])

        return objects
    else:
        return None

