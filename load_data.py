import os
import numpy as np
import matplotlib


files_read = 0

def load_batch_npz(data_folder,batch_size,height,width,channels,file_count,shuffle):
    global files_read

    inputs = np.empty((batch_size,height,width,channels),dtype=np.float32)
    labels = np.empty((batch_size,height,width,1),dtype=np.float32)
    if shuffle:
        premutes = np.random.permutation(np.arange(file_count))
    else:
        premutes = np.arange(file_count)

    for bi in range(batch_size):
        filename = data_folder + os.sep + 'res_2_img_%d.npz'%premutes[files_read]
        #print('loading file %s',filename)
        npzdata = np.load(filename)

        single_arr = np.reshape(npzdata['x_mat'],(height,width,1))
        single_arr = np.append(single_arr,np.reshape(npzdata['y_mat'],(height,width,1)),axis=2)
        inputs[bi,:,:,:] = single_arr
        labels[bi,:,:] = np.reshape(npzdata['image_mat'],(height,width,1))
        files_read = (files_read + 1)%file_count
        if files_read==0 and shuffle:
            premutes = np.random.permutation(np.arange(file_count))

    return inputs,labels
