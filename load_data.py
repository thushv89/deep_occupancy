import os
import numpy as np
import matplotlib.pyplot as plt
import sys


files_read = 0

def load_batch_npz(data_folder,batch_size,height,width,channels,file_count,shuffle,rand_cover_percentage,flip_lr,flip_ud,exp_type):
    """
    :param data_folder:
    :param batch_size:
    :param height:
    :param width:
    :param channels:
    :param file_count:
    :param shuffle:
    :param rand_cover_percentage: Size of the mask. [0, 1]. Around 0.5 is recommended. Ignore covering if None.
    :return:
    """
    global files_read

    inputs = np.empty((batch_size,height,width,channels),dtype=np.float32)
    labels = np.empty((batch_size,height,width,1),dtype=np.float32)
    if shuffle:
        premutes = np.random.permutation(np.arange(file_count))
    else:
        premutes = np.arange(file_count)

    for bi in range(batch_size):
        if exp_type != 'intel':
            filename = data_folder + os.sep + 'res_2_img_%d.npz'%premutes[files_read]
        else:
            filename = data_folder + os.sep + 'res_0.5_img_%d.npz' % premutes[files_read]
        #print('loading file %s',filename)
        npzdata = np.load(filename)
        image_mat = npzdata['image_mat']
        x_mat,y_mat = npzdata['x_mat'],npzdata['y_mat']
        if flip_lr and np.random.rand()<0.5:
            x_mat = np.fliplr(x_mat)
            y_mat = np.fliplr(y_mat)
            image_mat = np.fliplr(image_mat)

        if flip_ud and np.random.rand()<0.5:
            x_mat = np.flipud(x_mat)
            y_mat = np.flipud(y_mat)
            image_mat = np.flipud(image_mat)

        if rand_cover_percentage is not None:

            # generate a mask with a size < rand_cover_percentage of the image size
            mask = np.zeros(image_mat.shape)
            dx = np.random.randint(0, np.int(rand_cover_percentage*mask.shape[0]))
            x_init = np.random.randint(0, mask.shape[0]-dx)
            dy = np.random.randint(0, np.int(rand_cover_percentage*mask.shape[1]))
            y_init = np.random.randint(0, mask.shape[1]-dy)

            # 9 for unoccupied, 10 for unknown, and 11 for occupied
            mask[np.ix_(np.arange(x_init,x_init+dx), np.arange(y_init,y_init+dy))] = 10
            image_mat += mask

        single_arr = np.reshape(x_mat,(height,width,1))
        single_arr = np.append(single_arr,np.reshape(y_mat,(height,width,1)),axis=2)
        inputs[bi,:,:,:] = single_arr
        labels[bi,:,:] = np.reshape(image_mat,(height,width,1))
        files_read = (files_read + 1)%file_count
        if files_read==0 and shuffle:
            premutes = np.random.permutation(np.arange(file_count))

    return inputs,labels
