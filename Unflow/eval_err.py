import numpy as np
import matplotlib.pyplot as plt

inpath_ff_err = '/home/zhang/odo_err.txt'
angle_err = []
t_x_err = []
t_y_err = []
count = 0
with open(inpath_ff_err, 'r') as f:
    for line in f:
        angle_err.append([float(line.strip().split(' ')[0]) / np.pi *180, count])
        t_x_err.append([float(line.strip().split(' ')[1]), count])
        t_y_err.append([float(line.strip().split(' ')[2]), count])
        count += 1

angle_err = np.transpose(angle_err)
t_x_err = np.transpose(t_x_err)
t_y_err = np.transpose(t_y_err)

print(np.max(angle_err[0,:]), " " ,np.min(angle_err[0,:]), " ", np.sum(angle_err[0,:]) / count)
print(np.max(t_x_err[0,:]), " " ,np.min(t_x_err[0,:]), " ", np.sum(t_x_err[0,:]) / count)
print(np.max(t_y_err[0,:]), " " ,np.min(t_y_err[0,:]), " ", np.sum(t_y_err[0,:]) / count)



plt.figure(1)
ax1 = plt.subplot(311)
ax1.set_title("Relative rotation error")
ax1.set_xlabel('Timestamp')
ax1.set_ylabel('Degree')
ax1.plot(angle_err[1], angle_err[0], 'r--')

ax2 = plt.subplot(312)
ax2.set_xlabel('Timestamp')
ax2.set_ylabel('Meter')
ax2.set_title("Relative translation error in X direction")
ax2.plot(t_x_err[1], t_x_err[0], 'r--')

ax3 = plt.subplot(313)
ax3.set_xlabel('Timestamp')
ax3.set_ylabel('Meter')
ax3.set_title("Relative translation error in Y direction")
ax3.plot(t_y_err[1], t_y_err[0], 'r--')

plt.tight_layout(pad=1.0, w_pad=0.0, h_pad=-2.0)
plt.show()
