import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def motion_eval(labels_crt, labels_nxt, image_results, data_num, err, track_seq):
    img_gray = image_results[0]
    img_array = image_results[1]
    flow_u = image_results[2]
    flow_v = image_results[3]

    height_flow = img_gray.shape[1]
    width_flow = img_gray.shape[2]

    num = track_seq[0]
    grid_map_path = '/home/zhang/pcl_data/' + num + '/gridmap/'
    file = grid_map_path + str(data_num).zfill(6) + '_detections_cartesian.png'
    im_crt = Image.open(file).convert('L')
    im_crt.thumbnail((width_flow, height_flow), Image.ANTIALIAS)
    im_crt = np.array(im_crt).flatten()

    file = grid_map_path + str(data_num + 1).zfill(6) + '_detections_cartesian.png'
    im_nxt = Image.open(file).convert('RGB')
    im_nxt.thumbnail((width_flow, height_flow), Image.ANTIALIAS)

    if labels_crt is not None and labels_nxt is not None:
        for labels in labels_crt:
            [id_c, type_c, p_ul, p_ur, p_dl, p_dr] = labels

            img = Image.new("1", [width_flow, height_flow], 0)
            poly = p_ul + p_dl + p_dr + p_ur

            ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)

            img = np.array(img).flatten()
            mask = np.logical_and(img>0, im_crt>0)

            number_of_points = np.count_nonzero(mask)
            labels_nxt.append(['crt_' + id_c, type_c, p_ul, p_ur, p_dl, p_dr])

            if number_of_points != 0:
                flow_u_mask = np.dot(flow_u, mask)
                flow_u_avg = np.sum(flow_u_mask) / number_of_points
                flow_v_mask = np.dot(flow_v, mask)
                flow_v_avg = np.sum(flow_v_mask) / number_of_points
                [p_ul, p_ur, p_dl, p_dr] = add_motion([p_ul, p_ur, p_dl, p_dr], [flow_u_avg, flow_v_avg])
                labels_nxt.append(['est_'+id_c, type_c, p_ul, p_ur, p_dl, p_dr])

            for labels in labels_nxt:
                id_nxt = labels[0]
                if id_c == id_nxt:
                    mid_c = cal_mid([p_ul, p_ur, p_dl, p_dr])
                    mid_nxt = cal_mid(labels[2:6])
                    err.append([mid_c[0] - mid_nxt[0], mid_c[1] - mid_nxt[1]])

        for labels in labels_nxt:
            [id_c, type_c, p_ul, p_ur, p_dl, p_dr] = labels
            poly = p_ul + p_dl + p_dr + p_ur
            draw = ImageDraw.Draw(im_nxt)
            time = id_c.split('_')[0]
            if time == 'est':
                color = (0,255,0)
            elif time == 'crt':
                color = (0,0,255)
            else:
                color = (255,0,0)
            draw.polygon(poly, outline=color)
            draw.text(p_ul, type_c + id_c, color)

    im_nxt.save('/home/zhang/track/' + num + '/' + str(data_num).zfill(6) + '.png')
    return err


def add_motion(coord, flow):
    outputs = []
    for corners in coord:
        outputs.append([corners[0] + flow[0], corners[1] + flow[1]])
    return outputs


def cal_mid(coord):
    mid = [0.0, 0.0]
    for corners in coord:
        mid[0] = mid[0] + float(corners[0]) / 4.0
        mid[1] = mid[1] + float(corners[1]) / 4.0
    return mid