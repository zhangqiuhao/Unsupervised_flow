import numpy as np
import matplotlib.pyplot as plt

def eval_err():
    inpath_ff_err = '/home/zhang/odo_err.txt'
    angle_err = []
    t_x_err = []
    t_y_err = []
    count = 0
    with open(inpath_ff_err, 'r') as f:
        for line in f:
            angle_err.append([float(line.strip().split(' ')[0]) / np.pi * 180, count])
            t_x_err.append([float(line.strip().split(' ')[1]), count])
            t_y_err.append([float(line.strip().split(' ')[2]), count])
            count += 1

    angle_err = np.transpose(angle_err)
    t_x_err = np.transpose(t_x_err)
    t_y_err = np.transpose(t_y_err)

    plt.figure(1)
    ax1 = plt.subplot(411)
    ax1.set_title("Relative rotation error")
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Degree')
    ax1.plot(angle_err[1], angle_err[0], 'r--')

    ax2 = plt.subplot(412)
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Meter')
    ax2.set_title("Relative translation error in X direction")
    ax2.plot(t_x_err[1], t_x_err[0], 'r--')

    ax3 = plt.subplot(413)
    ax3.set_xlabel('Timestamp')
    ax3.set_ylabel('Meter')
    ax3.set_title("Relative translation error in Y direction")
    ax3.plot(t_y_err[1], t_y_err[0], 'r--')

    table = plt.subplot(414)

    data = [[round(np.max(angle_err[0, :]), 5), round(np.min(angle_err[0, :]), 5), round(np.sum(angle_err[0, :]) / count, 5)],
            [round(np.max(t_x_err[0, :]), 5), round(np.min(t_x_err[0, :]), 5), round(np.sum(t_x_err[0, :]) / count, 5)],
            [round(np.max(t_y_err[0, :]), 5), round(np.min(t_y_err[0, :]), 5), round(np.sum(t_y_err[0, :]) / count, 5)]]
    columns = ('Max_err', 'Min_err', 'Avg_err')
    rows = ['Angle', 'X Axis', 'Y Axis']

    cell_text = data
    # Add a table at the bottom of the axes
    table.axis("off")
    the_table = table.table(cellText=cell_text,
                            rowLabels=rows,
                            colLabels=columns,
                            loc='center')
    the_table.scale(1, 2)

    plt.subplots_adjust(left=0.2, bottom=0.1, hspace=0.1)
    plt.tight_layout(pad=1.0, w_pad=0.0, h_pad=0.0)
    plt.savefig('/home/zhang/odo_err.png')
