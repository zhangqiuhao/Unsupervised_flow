import numpy as np
import os
from PIL import Image
import os
import math
from matplotlib.colors import hsv_to_rgb

from Matrix import Matrix


def evaluate(file, file_err, old_R, old_t, matrix_input, num_iters, image_results , mode):
    img_gray = image_results[0]
    img_array = image_results[1]
    flow_u_array = image_results[2]
    flow_v_array = image_results[3]

    height = img_gray.shape[1]
    width = img_gray.shape[2]

    #create uv image and mask
    mask_array_non_zero = np.logical_and(np.logical_and(abs(flow_u_array) > 0.001, abs(flow_v_array) > 0.001), img_array>0.001)
    img_u = np.zeros((height, width))
    img_v = np.zeros((height, width))
    for v in range(height):
        img_v[v,:] =+ height - v
    for u in range(width):
        img_u[:,u] =+ u - width / 2.0
    img_v = img_v.flatten()
    img_u = img_u.flatten()

    matrix_flow_v = img_v + flow_v_array
    matrix_flow_u = img_u - flow_u_array

    matrix_im1 = []
    matrix_im2 = []

    boundry_pixels = 30
    mask_out_boundry = np.zeros((height, width), dtype = bool)
    mask_out_boundry[boundry_pixels:height - boundry_pixels, boundry_pixels:width-boundry_pixels] = True
    mask_out_boundry = mask_out_boundry.flatten()

    for idx, val in enumerate(np.logical_and(mask_array_non_zero,mask_out_boundry)):
        if val:
            matrix_im1.append([matrix_flow_v[idx], matrix_flow_u[idx], 0])
            matrix_im2.append([img_v[idx], img_u[idx], 0])

    matrix_im1 = np.transpose(np.asarray(matrix_im1))
    matrix_im2 = np.transpose(np.asarray(matrix_im2))

    print("Number of points: ", matrix_im1.shape[1])

    R,c,t = _ralign(matrix_im1, matrix_im2)

    #read now and previous groundtruth
    matrix_gt = matrix_input.return_matrix(num_iters + 3)
    matrix_gt_pr = matrix_input.return_matrix(num_iters + 2)
    R_gt = matrix_gt[:,0:3]
    t_gt = matrix_gt[:,3]
    R_gt_pr = matrix_gt_pr[:,0:3]
    t_gt_pr = matrix_gt_pr[:,3]
    R_gt = _rotationMatrixToEulerAngles(R_gt)[1]
    R_gt_pr = _rotationMatrixToEulerAngles(R_gt_pr)[1]

    delta_R_is = R_gt - R_gt_pr
    delta_t_x = t_gt[2] - t_gt_pr[2]
    delta_t_y = -(t_gt[0] - t_gt_pr[0])

    t_x = -(t[0] * np.cos(old_R) - t[1] * np.sin(old_R))
    t_y = t[0] * np.sin(old_R) + t[1] * np.cos(old_R)

    R_now = - _rotationMatrixToEulerAngles(R)[2]

    #print("delta R ", delta_R_is, " delta t_x ", delta_t_x, " delta t_y ", delta_t_y)
    R_err = R_now - delta_R_is
    t_x_err = t_x - delta_t_x.item(0)
    t_y_err = t_y - delta_t_y.item(0)
    #print("R_err: ", R_err, " t_x_err: ", t_x_err, " t_y_err: ", t_y_err, " c: ", c)

    R = old_R + R_now
    t_rotated = [(t_x+old_t[0]).item(0) , (t_y+old_t[1]).item(0)]
    file.write("0 0 0 " + str(t_rotated[0]) + " 0 0 0 " + str(t_rotated[1]) + " 0 " + str(R_now) + " " + str(t[0]) + " " + str(t[1]) + "\n")
    file_err.write(str(R_err) + " " + str(t_x_err) +  " " + str(t_y_err) + "\n")

    output_flow = np.zeros([height, width, 2])
    #Save flow without estimated self movement
    if mode == "estimated":
        t_1 = t[1] * 10.0
        t_0 = t[0] * 10.0
        flow_u_without = (flow_u_array + t_1 + img_u * np.cos(R_now) + img_v * np.sin(R_now) - img_u) * mask_array_non_zero
        flow_v_without = (flow_v_array + t_0 + img_u * np.sin(R_now) - img_v * np.cos(R_now) + img_v) * mask_array_non_zero
    elif mode == 'real':
    #Save flow without real self movement
        t_0 = (-(delta_t_x.item(0) * np.cos(old_R) - delta_t_y.item(0) * np.sin(old_R))) * 10.0
        t_1 = (delta_t_x.item(0) * np.sin(old_R) + delta_t_y.item(0) * np.cos(old_R)) * 10.0
        flow_u_without = (flow_u_array + t_1 + img_u * np.cos(delta_R_is) + img_v * np.sin(delta_R_is) - img_u) * mask_array_non_zero
        flow_v_without = (flow_v_array + t_0 + img_u * np.sin(delta_R_is) - img_v * np.cos(delta_R_is) + img_v) * mask_array_non_zero
    else:
        print('Should choose between real and estimated')

    output_flow[:,:,0] = flow_u_without.reshape((height, width))
    output_flow[:,:,1] = flow_v_without.reshape((height, width))

    img_flow_rgb = (_convert_flow_to_rgb(output_flow) * 255.0).astype(np.uint8)
    img_flow_rgb = Image.fromarray(img_flow_rgb)

    directory = '/home/zhang/flow_without_' + mode + '_motion/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    img_flow_rgb.save(directory + str(num_iters) + '_output_flow.jpeg')

    return R, t_rotated

def _convert_flow_to_rgb(flow):
    n = 2
    max_flow = 20
    flow_u = flow[:,:,0]
    flow_v = flow[:,:,1]
    mag = np.sqrt(np.sum(np.square(flow), axis = 2))
    angle = np.arctan2(flow_v, flow_u)

    im_h = np.mod(angle / (2 * np.pi) + 1.0, 1.0)
    im_s = np.clip(mag * n / max_flow, 0, 1)
    im_v = np.clip(n - im_s, 0, 1)

    im_hsv = np.zeros((flow.shape[0], flow.shape[1], 3))
    im_hsv[:,:, 0] = im_h
    im_hsv[:,:, 1] = im_s
    im_hsv[:,:, 2] = im_v
    im = hsv_to_rgb(im_hsv)

    return im

def _ralign(X,Y):
    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)
    Xc =  X - np.tile(mx, (n, 1)).T
    Yc =  Y - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(Xc*Xc, 0))
    sy = np.mean(np.sum(Yc*Yc, 0))

    Sxy = np.dot(Yc, Xc.T) / n

    U,D,V = np.linalg.svd(Sxy,full_matrices=True,compute_uv=True)
    V=V.T.copy()
    #print U,"\n\n",D,"\n\n",V
    r = np.rank(Sxy)
    d = np.linalg.det(Sxy)
    S = np.eye(m)
    if r > (m - 1):
        if ( np.det(Sxy) < 0 ):
            S[m, m] = -1;
        elif (r == m - 1):
            if (np.det(U) * np.det(V) < 0):
                S[m, m] = -1
        else:
            R = np.eye(2)
            c = 1
            t = np.zeros(2)
            return R,c,t

    R = np.dot( np.dot(U, S ), V.T)

    c = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - c * np.dot(R, mx)

    return R,c,t*0.1


# Checks if a matrix is a valid rotation matrix.
def _isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def _rotationMatrixToEulerAngles(R):
    assert (_isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])
