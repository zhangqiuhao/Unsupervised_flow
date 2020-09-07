import numpy as np
import os
from PIL import Image, ImageDraw
import os
import math
from matplotlib.colors import hsv_to_rgb
import random


SCALE_FACTOR = 0.1 * 600 / 512


def evaluate(file, file_err, file_kitti, old_R, old_t, matrix_input, num_iters, image_results, mode):
    img_gray = image_results[0]
    img_array = image_results[1]
    flow_u_array = image_results[2]
    flow_v_array = image_results[3]

    height = img_gray.shape[1]
    width = img_gray.shape[2]

    #create uv image and mask
    mask_array_non_zero = np.logical_and(np.logical_and(abs(flow_u_array) > 0.001, abs(flow_v_array) > 0.001),
                                         img_array > 0.001)

    boundry_pixels = 50
    mask_out_boundry = np.zeros((height, width), dtype=bool)
    mask_out_boundry[boundry_pixels:height - boundry_pixels, boundry_pixels:width-boundry_pixels] = True
    mask_out_boundry = mask_out_boundry.flatten()
    mask = np.logical_and(mask_array_non_zero, mask_out_boundry)

    img_z, img_x, R_center = _calculate_rotation_midpoint(height, width, flow_v_array, flow_u_array, mask)

    matrix_flow_z = img_z - flow_v_array
    matrix_flow_x = img_x + flow_u_array

    matrix_im1 = []
    matrix_im2 = []

    for idx, val in enumerate(mask):
        if val:
            matrix_im1.append([img_x[idx], 0, img_z[idx]])
            matrix_im2.append([matrix_flow_x[idx], 0, matrix_flow_z[idx]])

    matrix_im1 = np.transpose(np.asarray(matrix_im1))
    matrix_im2 = np.transpose(np.asarray(matrix_im2))

    R, t = weighted_ralign(matrix_im2, matrix_im1)
    R_now = -rotationMatrixToEulerAngles(R)[1]
    t = [t[2]*np.tan(R_now), t[2]]
    rot_mat = np.linalg.inv(np.array([[np.cos(old_R), -np.sin(old_R)], [np.sin(old_R), np.cos(old_R)]]))
    t_rot = np.dot(rot_mat, t)
    t_pose = t_rot + old_t

    if matrix_input is not None:
        delta_t_x, delta_t_y, delta_R_is = calculate_err(matrix_input, num_iters, R_now, t, file, file_err, t_rot[0], t_rot[1], t_pose)
    else:
        mode = 'estimated'

    output_flow = np.zeros([height, width, 2])
    #Save flow without estimated self movement
    flow_x_without=[]
    flow_z_without=[]
    if mode == "estimated":
        t_0 = t[0] / SCALE_FACTOR
        t_1 = t[1] / SCALE_FACTOR
        flow_x_without = (flow_u_array - (t_0 + img_x * np.cos(R_now) - img_z * np.sin(R_now) - img_x)) * mask_array_non_zero
        flow_z_without = (flow_v_array + (t_1 + img_x * np.sin(R_now) + img_z * np.cos(R_now) - img_z)) * mask_array_non_zero
    elif mode == 'real':
    #Save flow without real self movement
        t_0 = (-(delta_t_x.item(0) * np.cos(old_R) - delta_t_y.item(0) * np.sin(old_R))) / SCALE_FACTOR
        t_1 = (delta_t_x.item(0) * np.sin(old_R) + delta_t_y.item(0) * np.cos(old_R)) / SCALE_FACTOR
        flow_x_without = (flow_u_array - (t_1 + img_x * np.cos(delta_R_is) - img_z * np.sin(delta_R_is) - img_x)) * mask_array_non_zero
        flow_z_without = (flow_v_array - (t_0 + img_x * np.sin(delta_R_is) + img_z * np.cos(delta_R_is) - img_z)) * mask_array_non_zero
    else:
        print('Should choose between real and estimated')

    #write image and odometry
    #odo_file
    file.write("0 0 0 " + str(t_pose[0]) + " 0 0 0 " + str(t_pose[1]) + " 0 " + str(R_now) + " " + str(t[0]) + " " + str(t[1]) + "\n")
    angle_text = ''

    #kitti_odo
    rot_mat = Euler_angle_to_rot_mat([0.0, old_R, 0.0])
    kitti_t = [old_t[0], 0.0, old_t[1]]
    for row, mat_row in enumerate(rot_mat):
        for text in mat_row:
            angle_text += str(text) + ' '
        angle_text += str(kitti_t[row]) + ' '
    file_kitti.write(angle_text + "\n")

    output_flow[:, :, 0] = flow_x_without.reshape((height, width))
    output_flow[:, :, 1] = flow_z_without.reshape((height, width))

    img_flow_rgb = (convert_flow_to_rgb(output_flow) * 255.0).astype(np.uint8)
    img_flow_rgb = Image.fromarray(img_flow_rgb)

    directory = '/home/zqhyyl/flow_without_' + mode + '_motion/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    img_flow_rgb.save(directory + str(num_iters) + '_output_flow.jpeg')

    R = old_R + R_now
    return R, t_pose


def calculate_err(matrix_input, num_iters, R_now, t, file, file_err, t_x, t_y, t_rotated):
    #read now and previous groundtruth
    matrix_gt = matrix_input.return_matrix(num_iters + 1)
    matrix_gt_pr = matrix_input.return_matrix(num_iters)
    R_gt = matrix_gt[:,0:3]
    t_gt = matrix_gt[:,3]
    R_gt_pr = matrix_gt_pr[:,0:3]
    t_gt_pr = matrix_gt_pr[:,3]

    R_gt = rotationMatrixToEulerAngles(R_gt)[1]
    R_gt_pr = rotationMatrixToEulerAngles(R_gt_pr)[1]

    delta_R_is = R_gt - R_gt_pr
    delta_t_x = t_gt[2] - t_gt_pr[2]
    delta_t_y = -(t_gt[0] - t_gt_pr[0])

    R_err = R_now - delta_R_is
    t_x_err = t_x - delta_t_x.item(0)
    t_y_err = t_y - delta_t_y.item(0)

    file_err.write(str(R_err) + " " + str(t_x_err) + " " + str(t_y_err) + "\n")
    return delta_t_x, delta_t_y, delta_R_is


def convert_flow_to_rgb(flow):
    n = 2
    max_flow = 20
    flow_u = flow[:,:,0]
    flow_v = flow[:,:,1]
    mag = np.sqrt(np.sum(np.square(flow), axis=2))
    angle = np.arctan2(flow_v, flow_u)

    im_h = np.mod(angle / (2 * np.pi) + 1.0, 1.0)
    im_s = np.clip(mag * n / max_flow, 0, 1)
    im_v = np.clip(n - im_s, 0, 1)

    im_hsv = np.zeros((flow.shape[0], flow.shape[1], 3))
    im_hsv[:, :, 0] = im_h
    im_hsv[:, :, 1] = im_s
    im_hsv[:, :, 2] = im_v
    im = hsv_to_rgb(im_hsv)

    return im


def weighted_ralign(old_points, new_points, number_iterations=10):
    n_ = np.ones_like(old_points[0, :])
    weights = np.diag(n_)
    vec_1_n = np.expand_dims(n_, 0)

    for i in range(number_iterations):
        sum_p = np.sum(weights)
        mx = np.dot(np.dot(old_points, weights), n_) / sum_p
        my = np.dot(np.dot(new_points, weights), n_) / sum_p

        Xc = old_points - np.dot(np.expand_dims(mx,1), vec_1_n)
        Yc = new_points - np.dot(np.expand_dims(my,1), vec_1_n)

        A = np.dot(np.dot(Xc, weights), Yc.T) / sum_p
        U,D,V = np.linalg.svd(A, full_matrices=True, compute_uv=True)

        ones = np.ones_like(old_points[:, 0])
        S = np.diag(ones)

        R = np.dot(np.dot(U, S), V)
        t = my.T - np.dot(R, mx.T)
        weights = squared_errors_to_weights(R, t, old_points, new_points)
    return R, t*SCALE_FACTOR


def squared_errors_to_weights(R, t, old_points, new_points):
    t = np.expand_dims(t, 1)
    err = np.sum(np.square(np.add(np.dot(R, old_points), t) - new_points), axis=0)
    max = np.clip(np.max(err), a_min=1e-30, a_max=1e100)
    weights = (max-err) / max
    weights = np.diag(weights)
    return weights


def _ralign(old_points, new_points):
    A = np.dot(old_points, new_points.T)
    U, D, V = np.linalg.svd(A, full_matrices=True, compute_uv=True)
    mx = np.mean(old_points, axis=1)
    my = np.mean(new_points, axis=1)
    ones = np.ones_like(old_points[:, 0])
    S = np.diag(ones)
    R = np.dot(np.dot(U, S), V.T)
    t = my - np.dot(R, mx)
    return R, 1.0, t*SCALE_FACTOR


# Checks if a matrix is a valid rotation matrix.
def _isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-5


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
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


def _calculate_rotation_midpoint(height, width, flow_v_array, flow_u_array, mask):
    flow_none_zeros = []
    for idx, val in enumerate(mask):
        if val:
            flow_none_zeros.append([flow_v_array[idx], flow_u_array[idx]])

    flow_none_zeros = np.asarray(flow_none_zeros)
    length = flow_none_zeros.shape[0]
    if length%2 != 0:
        length = length - 1
    flow_left = flow_none_zeros[0:int(length/2), :]
    flow_left = np.divide(flow_left, np.reshape(np.linalg.norm(flow_left, axis=-1), (int(length/2), 1)))
    flow_right = flow_none_zeros[int(length/2):length, :]
    flow_right = np.divide(flow_right, np.reshape(np.linalg.norm(flow_right, axis=-1), (int(length/2), 1)))

    check_parallel = (np.abs(np.cross(flow_left, flow_right)) > 0.1).sum() / int(length/2)

    img_u = np.zeros((height, width))
    img_v = np.zeros((height, width))
    for v in range(height):
        img_v[v, :] = v
    for u in range(width):
        img_u[:, u] = u
    if check_parallel > 1.0:
        print("Percentage:" + str(check_parallel) + " Estimating rotation midpoint")
        img_v_flat = np.asarray(img_v.flatten())
        img_u_flat = np.asarray(img_u.flatten())
        Y = []
        img_VU = []
        for idx, val in enumerate(mask):
            if val:
                Y.append(flow_v_array[idx] * img_v_flat[idx] + flow_u_array[idx] * img_u_flat[idx])
                img_VU.append([img_v_flat[idx], img_u_flat[idx]])
        center = center_ransac(flow_none_zeros, Y, img_VU)
        print(center)
    else:
        center = [height+10, width / 2.0]
    img_z = center[0] - img_v
    img_x = img_u - center[1]
    return img_z.flatten(), img_x.flatten(), center


def iter_ransac(X, Y, all_X, all_Y, iterates=5, stop_at_goal=True):
    n = all_X.shape[1]
    print("Number of points: ", n)
    goal_inliers = n * 0.9
    best_ic = 0
    best_model = None

    for iter_num in range(iterates):
        model, ic, X, Y = ransac(X, Y, all_X, all_Y)

        if ic > best_ic:
            best_ic = ic
            best_model = model
            if ic > goal_inliers and stop_at_goal:
                break
    return best_model


def ransac(X, Y, all_X, all_Y, max_iterations=50, stop_at_goal=True, random_seed=None):
    n = all_X.shape[1]
    sample_size = X.shape[1]
    goal_inliers = n * 0.9

    best_ic = 0
    best_model = None
    best_X = []
    best_Y = []

    data = np.concatenate([X, Y], axis=0)
    data = data.T
    random.seed(random_seed)

    # random.sample cannot deal with "data" being a numpy array
    data = list(data)
    for i in range(max_iterations):
        s = random.sample(data, int(sample_size))
        s = np.asarray(s)
        array_1 = np.transpose(s[:, :3])
        array_2 = np.transpose(s[:, 3:])

        R, c, t = _ralign(array_1, array_2)
        ic, X, Y = count_inlier(R, c, t, all_X, all_Y)

        if ic > best_ic:
            best_ic = ic
            best_X = X
            best_Y = Y
            best_model = [R, c, t * SCALE_FACTOR]
            if ic > goal_inliers and stop_at_goal:
                break

    return best_model, ic, best_X, best_Y


def count_inlier(R, c, t, array_1, array_2):
    transformed_img = np.add(np.dot(R, array_1), np.reshape(t, (3, 1)))
    err = np.sum(np.square(np.subtract(transformed_img, array_2)), axis=0)
    small_err = np.where(err < np.min(np.mean(err)))
    X = []
    Y = []
    for i in small_err:
        X.append(array_1[:, i])
        Y.append(array_2[:, i])
    X = np.asarray(X)
    Y = np.asarray(Y)
    ic = len(small_err[0])
    return ic, X[0], Y[0]


def center_ransac(X, Y, img_VU, sample_size=50, max_iterations=100, stop_at_goal=True, random_seed=None):
    n = X.shape[0]
    print("Number of points: ", n)
    goal_inliers = n * 0.8

    best_ic = 0
    best_model = None

    Y = np.reshape(Y, (n, 1))
    data = np.concatenate([X, Y], axis=1)
    random.seed(random_seed)

    # random.sample cannot deal with "data" being a numpy array
    data = list(data)
    for i in range(max_iterations):
        s = random.sample(data, int(sample_size))
        s = np.asarray(s)
        array_1 = s[:, :2]
        array_2 = s[:, 2:]

        array_1_T = np.transpose(array_1)
        center = np.dot(np.dot(np.linalg.inv(np.dot(array_1_T, array_1)), array_1_T), array_2)
        ic = count_center_inlier(center, X, img_VU)

        if ic > best_ic:
            best_ic = ic
            best_model = center
            if ic > goal_inliers and stop_at_goal:
                break
    return best_model


def count_center_inlier(center, X, img_VU):
    normal_vector = np.subtract(img_VU, np.transpose(center))
    normal_vector = np.reshape(normal_vector, (normal_vector.shape[0], 1, 2))
    X = np.reshape(X, (X.shape[0], 2, 1))
    err = np.reshape(np.einsum('ipq,iqr->ipr', normal_vector, X), (X.shape[0]))
    ic = len(np.where(np.abs(err) < 10))
    print(ic)
    return ic


def Euler_angle_to_rot_mat(angles):
    [X, Y, Z] = angles
    ZMatrix = np.matrix([
        [math.cos(Z), -math.sin(Z), 0],
        [math.sin(Z), math.cos(Z), 0],
        [0, 0, 1]
    ])

    YMatrix = np.matrix([
        [math.cos(Y), 0, math.sin(Y)],
        [0, 1, 0],
        [-math.sin(Y), 0, math.cos(Y)]
    ])

    XMatrix = np.matrix([
        [1, 0, 0],
        [0, math.cos(X), -math.sin(X)],
        [0, math.sin(X), math.cos(X)]
    ])

    R = ZMatrix * YMatrix * XMatrix
    return np.asarray(R)
