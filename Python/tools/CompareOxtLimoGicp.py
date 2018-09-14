import sys, os
import numpy as np
from numpy.linalg import inv
import pcl
from numpngw import write_png
from math import degrees
import matplotlib.pyplot as plt
from PlotDirectionalField import directionalField
import svgwrite
from matplotlib.path import Path
import matplotlib.patches as patches
from NumpyHelpers import XYAngleToMatrix2, rotationMatrixToEulerAngles


def main(args):
    np.set_printoptions(suppress=True)

    try:
        sequence = int(args[1])
    except:
        sequence = 9

    inpath_tf_oxt = '/mrtstorage/datasets/kitti/odometry/all_py/poses/' + str(sequence).zfill(2) + '.txt'
    inpath_tf_limo = '/mrtstorage/datasets/kitti/odometry/limo_results/' + str(sequence).zfill(2) + '.txt'
    inpath_tf_gicp = '/home/klein/U/gridmapsfull/train/kitti' + str(sequence).zfill(2) + '/gridmaps.csv'


    outfolder = '/home/klein/U/kittitraj/'

    with open(inpath_tf_oxt, 'r') as f:
        oxt = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]
    with open(inpath_tf_limo, 'r') as f:
        limo = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]

    with open(inpath_tf_gicp, 'r') as f:
        gicp_raw = [[float(i) for i in line.split(',')[-3:]] for line in f]
    gicp_raw = [[0.0, 0.0, 0.0]] + gicp_raw

    i = 99999
    oxt = oxt[:i]
    limo = limo[:i]
    gicp_raw = gicp_raw[:i]

    plot_data_gicp = []
    points_gicp = []
    pos = np.identity(3)
    for tf in gicp_raw:
        pos = pos @ XYAngleToMatrix2(tf)
        points_gicp.append((pos[0,2]/10, -pos[1,2]/10))
        plot_data_gicp.append(tf)

    points_oxt = []
    plot_data_oxt = []
    lasttf = np.identity(4)
    for l in oxt:
        points_oxt.append((l[2, 3], -l[0, 3]))
        diff = inv(lasttf) @ l
        plot_data_oxt.append([diff[2, 3], -diff[0, 3], -degrees(rotationMatrixToEulerAngles(diff[0:3,0:3])[1])])
        lasttf = l

    points_limo = []
    plot_data_limo = []
    lasttf = np.identity(4)
    for l in limo:
        points_limo.append((l[2,3], -l[0,3]))
        diff = inv(lasttf) @ l
        plot_data_limo.append([diff[2, 3], -diff[0, 3], -degrees(rotationMatrixToEulerAngles(diff[0:3,0:3])[1])])
        lasttf = l

    path_oxt = Path(points_oxt, [Path.MOVETO] + [Path.LINETO] * (len(points_oxt) - 1))
    path_limo = Path(points_limo, [Path.MOVETO] + [Path.LINETO] * (len(points_limo)-1))
    path_gicp = Path(points_gicp, [Path.MOVETO] + [Path.LINETO] * (len(points_gicp) - 1))

    # trajectory plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.title('Trajectory [m, m]')
    # ax.legend(handles=[patches.Patch(color='C0', label='OXT'), patches.Patch(color='C1', label='LIMO'), patches.Patch(color='C2', label='GICP')])
    ax.add_patch(patches.PathPatch(path_oxt, facecolor='none', edgecolor='C0'))
    # ax.add_patch(patches.PathPatch(path_limo, facecolor='none', edgecolor='C1'))
    # ax.add_patch(patches.PathPatch(path_gicp, facecolor='none', edgecolor='C2'))
    ax.set_aspect('equal', 'datalim')
    plt.autoscale(enable=True)

    # # velocity plots
    # fig2 = plt.figure()
    #
    # plot_data_gicp = np.array(plot_data_gicp) *  [0.1, 0.1, 1]
    # plot_data_oxt = np.array(plot_data_oxt)
    # plot_data_limo = np.array(plot_data_limo)
    #
    # ax = fig2.add_subplot(311)
    # plt.title('X-Translation [steps, m]')
    # ax.legend(handles=[patches.Patch(color='C0', label='OXT'), patches.Patch(color='C1', label='LIMO'),
    #                    patches.Patch(color='C2', label='GICP')])
    # steps = range(len(plot_data_gicp))
    # plt.plot(steps, plot_data_oxt[:,0], steps, plot_data_limo[:,0], steps, plot_data_gicp[:,0])
    #
    # ax = fig2.add_subplot(312)
    # plt.title('Y-Translation [steps, m]')
    # ax.legend(handles=[patches.Patch(color='C0', label='OXT'), patches.Patch(color='C1', label='LIMO'),
    #                    patches.Patch(color='C2', label='GICP')])
    # steps = range(len(plot_data_gicp))
    # plt.plot(steps, plot_data_oxt[:,1], steps, plot_data_limo[:,1], steps, plot_data_gicp[:,1])
    #
    # ax = fig2.add_subplot(313)
    # plt.title('Angle Difference [steps, deg]')
    # ax.legend(handles=[patches.Patch(color='C0', label='OXT'), patches.Patch(color='C1', label='LIMO'),
    #                    patches.Patch(color='C2', label='GICP')])
    # steps = range(len(plot_data_gicp))
    # plt.plot(steps, plot_data_oxt[:,2], steps, plot_data_limo[:,2], steps, plot_data_gicp[:,2])
    #
    # outfolder = '/home/klein/U/comparison_oxt_limo_gicp/'
    # if not os.path.exists(outfolder):
    #     os.makedirs(outfolder)

    fig.savefig(outfolder + 'kitti_'+str(sequence).zfill(2)+'_trajectory.png')
    # fig2.savefig(outfolder + 'kitti_' + str(sequence).zfill(2) + '_xyangle.png')

    # plt.show()



if __name__ == "__main__":
    main(sys.argv)
