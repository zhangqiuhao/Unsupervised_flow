import numpy as np
from math import sin,cos,radians,degrees,acos,sqrt,atan2


def XYAngleToMatrix(x, y, angle):
    """Returns a numpy array containing the 2D transformation matrix from
    the given coordinates x,y and angle in degrees."""
    return np.array([ [cos(radians(angle)),  sin(radians(angle)), x],
                      [-sin(radians(angle)), cos(radians(angle)), y],
                      [0, 0, 1]], dtype=np.float32)


def XYAngleToMatrix2(data):
    """Returns a numpy array containing the 2D transformation matrix from
    the given data which holds coordinates x,y and angle in degrees."""
    return np.array([ [cos(radians(data[2])),  sin(radians(data[2])), data[0]],
                      [-sin(radians(data[2])), cos(radians(data[2])), data[1]],
                      [0, 0, 1]], dtype=np.float32)


def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    #assert (isRotationMatrix(R))

    sy = sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = atan2(R[2, 1], R[2, 2])
        y = atan2(-R[2, 0], sy)
        z = atan2(R[1, 0], R[0, 0])
    else:
        x = atan2(-R[1, 2], R[1, 1])
        y = atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def similarity_transform(from_points, to_points):
    assert len(from_points.shape) == 2, \
        "from_points must be a m x n array"
    assert from_points.shape == to_points.shape, \
        "from_points and to_points must have the same shape"

    N, m = from_points.shape

    mean_from = from_points.mean(axis=0)
    mean_to = to_points.mean(axis=0)

    delta_from = from_points - mean_from  # N x m
    delta_to = to_points - mean_to  # N x m

    sigma_from = (delta_from * delta_from).sum(axis=1).mean()
    sigma_to = (delta_to * delta_to).sum(axis=1).mean()

    cov_matrix = delta_to.T.dot(delta_from) / N

    U, d, V_t = np.linalg.svd(cov_matrix, full_matrices=True)
    cov_rank = np.linalg.matrix_rank(cov_matrix)
    S = np.eye(m)

    if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
        S[m - 1, m - 1] = -1
    elif cov_rank < m - 1:
        raise ValueError("colinearity detected in covariance matrix:\n{}".format(cov_matrix))

    R = U.dot(S).dot(V_t)
    c = (d * S.diagonal()).sum() / sigma_from
    t = mean_to - c * R.dot(mean_from)

    return c * R, t



def main(args):

    pointsA = np.random.rand(10,2)*4
    print(pointsA)

    # rotate by 10 degrees:
    phi = radians(2)
    pointsB = []
    for point in pointsA:
        pointsB.append(point @ np.array([[cos(phi), -sin(phi)],[sin(phi), cos(phi)]]))


    # shift by 10 in x direction
    pointsB = np.array(pointsB) + np.array([10, 5])

    print(pointsB)

    print()

    R, t = similarity_transform(pointsA, pointsB)
    print(degrees(acos(R[0,0])), t)

    return


if __name__ == '__main__':
    import sys
    main(sys.argv)