import tensorflow as tf
import os, sys
import numpy as np
sys.path.append('/home/klein/U/Masterarbeit/CUDAops/')
from flownet_ops.op_correlation import correlation
from flownet_ops.op_flow_warp import flow_warp

sess = tf.Session()

outfolder = '/home/klein/U/Masterarbeit/Python/FlowNet/extracted_weights/flownet_cs_kitti/'

if not os.path.exists(outfolder):
    os.makedirs(outfolder)

# new_saver = tf.train.import_meta_graph('/home/klein/U/Masterarbeit/FlowNet2/flownet2-tf/checkpoints/FlowNetC/flownet-C.ckpt-0.meta')
# what=new_saver.restore(sess, '/home/klein/U/Masterarbeit/FlowNet2/flownet2-tf/checkpoints/FlowNetC/flownet-C.ckpt-0')

new_saver = tf.train.import_meta_graph('/home/klein/U/net/flownet/activation_fixed/flownet_CS/model.ckpt-74848.meta')
what=new_saver.restore(sess, '/home/klein/U/net/flownet/activation_fixed/flownet_CS/model.ckpt-74848')


all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

# self trained flownets
for v in all_vars:

    if 'Adam' not in str(v):
        # print(v)
        if 'FlowNet_S/' in str(v) or 'FlowNet_C/' in str(v):
            name = '_'.join(str(v).split("'")[1].split('/')[0:2]).replace('conv2d', 'predict_2')#.replace('_S_','_S2_')

        # elif 'FlowNet_S_fixed/' in str(v) or 'FlowNet_C/' in str(v):
        #     name = str(v).split("'")[1].split('/')[1].replace('conv2d', 'predict_2')
        #
        #
        # if 'FlowNet_S/' in str(v) or 'FlowNet_C/' in str(v):
        #     name = '_'.join(str(v).split("'")[1].split('/')[0:2]).replace('conv2d', 'predict_2')

        if 'kernel' in str(v):
            print(name)

            v_ = np.array(sess.run(v))
            np.save(outfolder+name+'_weights', v_)

        if 'bias' in str(v):
            print(name)

            v_ = np.array(sess.run(v))
            np.save(outfolder+name+'_biases', v_)


# for own flownet C or S or SD
# for v in all_vars:
#
#     if 'Adam' not in str(v) and '/' in str(v):
#         print(v)
#         name = str(v).split("'")[1].split('/')[0].replace('conv2d', 'predict_2')
#         print(name)
#
#         if 'kernel' in str(v):
#             print(name)
#
#             v_ = np.array(sess.run(v))
#             np.save(outfolder+name+'_weights', v_)
#
#         if 'bias' in str(v):
#             print(name)
#
#             v_ = np.array(sess.run(v))
#             np.save(outfolder+name+'_biases', v_)


# flownet c and s
# for v in all_vars:
#
#     if 'weights' in str(v):
#         name = str(v).split("'")[1].split('/')[1]
#         print(name)
#
#         v_ = np.array(sess.run(v))
#         np.save(outfolder+name+'_weights', v_)
#
#     if 'biases' in str(v):
#         name = str(v).split("'")[1].split('/')[1]
#         print(name)
#
#         v_ = np.array(sess.run(v))
#         np.save(outfolder+name+'_biases', v_)


# flownet css
# for v in all_vars:
#
#     if 'FlowNetCSS/FlowNetS/' in str(v):
#
#         name = str(v).split("'")[1].split('/')[2]
#
#         print(name, '\n')
#         if 'weights' in str(v):
#
#             v_ = np.array(sess.run(v))
#             np.save(outfolder+name+'_weights', v_)
#
#         if 'biases' in str(v):
#
#             v_ = np.array(sess.run(v))
#             np.save(outfolder+name+'_biases', v_)



# flownet 2
# for v in all_vars:
#
#     if 'FlowNet2/' in str(v):
#
#         if not 'FlowNet2/Flow' in str(v):
#
#             name = str(v).split("'")[1].split('/')[1]
#             print(name)
#
#             if 'weights' in str(v):
#
#                 v_ = np.array(sess.run(v))
#                 np.save(outfolder+name+'_weights', v_)
#
#             if 'biases' in str(v):
#
#                 v_ = np.array(sess.run(v))
#                 np.save(outfolder+name+'_biases', v_)