from __future__ import division #python 2 division

import numpy as np
import tensorflow as tf
import load_data
import matplotlib.pyplot  as plt
import os
from skimage import io,img_as_uint
import sys
from scipy import ndimage

CONV_SCOPES = ['conv1','conv2','conv3','conv4','deconv2','deconv1']
# in 30,60
# after pool1 : 15,30
# after pool2 : 5,10
conv1kernel = 3
conv2kernel = 3

batch_size = 10
OUTPUT_TYPE = 'regression'

VAR_SHAPES = {'conv1': [conv1kernel,conv1kernel,2,16],'conv2': [1,1,16,16],
              'conv3':[conv2kernel,conv2kernel,16,32],'conv4': [1,1,32,32],
              'deconv2':[1,1,16,32],'deconv1':[1,1,1,16]}
VAR_REC_SHAPES = {'conv1': [conv1kernel,conv1kernel,1,16],'conv2': [1,1,16,16],
                  'conv3':[conv2kernel,conv2kernel,16,32],'conv4': [1,1,32,32],
                  'deconv2':[1,1,16,32],'deconv1':[1,1,1,16]}

OUTPUT_SHAPES = {'deconv2':[batch_size,30, 60,16],'deconv1':[batch_size,30,60,1]}
TEST_OUTPUT_SHAPES = {'deconv2':[1,30, 60,16],'deconv1':[1,30,60,1]}


POOL_STRIDES = {'pool1':[1,2,2,1],'pool2':[1,3,3,1]}
TF_WEIGHTS_SCOPE = 'weights'
TF_BIAS_SCOPE = 'bias'
TF_DECONV_SCOPE = 'deconv'
TF_RECONSTRUCTION_SCOPE = 'reconstruct'

ACTIVATION = 'lrelu'
beta = 0.00001
AUTO_DECREASE_LR = True
ACCURACY_DROP_CAP = 3

test_width = 600
test_height = 300
pred_data_dir = 'pred_data_rec'

normalize_constant = 150

graph = tf.get_default_graph()
sess = tf.InteractiveSession(graph=graph)


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def activate(x,activation_type,name='activation'):

    if activation_type=='tanh':
        return tf.nn.tanh(x)
    elif activation_type=='relu':
        return tf.nn.relu(x)
    elif activation_type=='lrelu':
        return lrelu(x)
    else:
        raise NotImplementedError


def build_tensorflw_variables():
    '''
    Build the required tensorflow variables to randomly initialize weights of the CNN
    :param variable_shapes: Shapes of the variables
    :return:
    '''
    global logger,sess,graph
    global CONV_SCOPES, VAR_SHAPES, TF_WEIGHTS_SCOPE, TF_BIAS_SCOPE, TF_DECONV_SCOPE

    print("Building Tensorflow Variables (Tensorflow)...")
    for si,scope in enumerate(CONV_SCOPES):
        with tf.variable_scope(scope):

            # Try Except because if you try get_variable with an intializer and
            # the variable exists, you will get a ValueError saying the variable exists
            try:
                if scope.startswith('conv'):
                    tf.get_variable(TF_WEIGHTS_SCOPE, shape=VAR_SHAPES[scope],
                                              initializer=tf.contrib.layers.xavier_initializer())
                    tf.get_variable(TF_BIAS_SCOPE,
                                           initializer = tf.random_uniform(shape=[VAR_SHAPES[scope][-1]],minval=-0.01,maxval=0.01,dtype=tf.float32))

                if scope.startswith('deconv'):
                    tf.get_variable(TF_WEIGHTS_SCOPE, shape=VAR_SHAPES[scope],
                                              initializer=tf.contrib.layers.xavier_initializer())
                    tf.get_variable(TF_BIAS_SCOPE,
                                           initializer=tf.random_uniform(shape=[VAR_SHAPES[scope][-2]], minval=-0.01,maxval=0.01,dtype=tf.float32))
            except ValueError as e:
                print(e)

        print([v.name for v in tf.global_variables()])



def build_tensorflw_reconstruction_variables():
    '''
    Build the required tensorflow variables to randomly initialize weights of the CNN
    :param variable_shapes: Shapes of the variables
    :return:
    '''
    global logger,sess,graph
    global CONV_SCOPES, VAR_REC_SHAPES, TF_WEIGHTS_SCOPE, TF_BIAS_SCOPE, TF_DECONV_SCOPE

    print("Building Tensorflow Variables (Tensorflow)...")
    for si,scope in enumerate(CONV_SCOPES):
        with tf.variable_scope(TF_RECONSTRUCTION_SCOPE):
            with tf.variable_scope(scope):
                # Try Except because if you try get_variable with an intializer and
                # the variable exists, you will get a ValueError saying the variable exists
                try:
                    if scope.startswith('conv'):
                        tf.get_variable(TF_WEIGHTS_SCOPE, shape=VAR_REC_SHAPES[scope],
                                        initializer=tf.contrib.layers.xavier_initializer())
                        tf.get_variable(TF_BIAS_SCOPE,
                                        initializer=tf.random_uniform(shape=[VAR_REC_SHAPES[scope][-1]], minval=-0.01,
                                                                      maxval=0.01, dtype=tf.float32))

                    if scope.startswith('deconv'):
                        tf.get_variable(TF_WEIGHTS_SCOPE, shape=VAR_REC_SHAPES[scope],
                                        initializer=tf.contrib.layers.xavier_initializer())
                        tf.get_variable(TF_BIAS_SCOPE,
                                        initializer=tf.random_uniform(shape=[VAR_REC_SHAPES[scope][-2]], minval=-0.01,
                                                                      maxval=0.01, dtype=tf.float32))
                except ValueError as e:
                    print(e)

        print([v.name for v in tf.global_variables()])


def get_inference(tf_inputs,OUTPUT_SHAPES):
    global TF_WEIGHTS_SCOPE,TF_BIAS_SCOPE,TF_DECONV_SCOPE,POOL_STRIDES
    # forward pass
    for si, scope in enumerate(CONV_SCOPES):
        with tf.variable_scope(scope, reuse=True) as sc:

            if scope.startswith('conv'):
                weight, bias = tf.get_variable(TF_WEIGHTS_SCOPE), tf.get_variable(TF_BIAS_SCOPE)
                print('\t\tConvolution with ReLU activation for ', scope)
                if si == 0:
                    print('\t\t\tInput shape ', tf_inputs.get_shape().as_list())
                    h = activate(
                        tf.nn.conv2d(tf_inputs, weight, strides=[1,1,1,1], padding='SAME') + bias,
                        activation_type=ACTIVATION, name='hidden')
                    print('\t\t\tOutput shape: ', h.get_shape().as_list())
                else:
                    h = activate(tf.nn.conv2d(h, weight, strides=[1,1,1,1], padding='SAME') + bias,
                                 activation_type=ACTIVATION, name='hidden')
                    print('\t\t\tOutput shape: ', h.get_shape().as_list())
            if scope.startswith('deconv'):
                weight, bias = tf.get_variable(TF_WEIGHTS_SCOPE), tf.get_variable(TF_BIAS_SCOPE)

                if si== len(CONV_SCOPES)-1:
                    if OUTPUT_TYPE == 'regression':
                        print('\t\tConvolution with TanH activation for ', scope)

                        h = tf.nn.tanh(tf.nn.conv2d_transpose(h, weight,OUTPUT_SHAPES[scope],strides=[1,1,1,1],padding="SAME") + bias)
                        print('\t\t\tOutput shape: ', h.get_shape().as_list())
                    elif OUTPUT_TYPE == 'classification':
                        print('\t\tConvolution with logits for ', scope)
                        h = tf.nn.conv2d_transpose(h, weight, OUTPUT_SHAPES[scope],
                                                   strides=[1, 1, 1, 1], padding="SAME") + bias
                        print('\t\t\tOutput shape: ', h.get_shape().as_list())

                else:
                    print('\t\tConvolution with ReLU activation for ', scope)
                    h = activate(tf.nn.conv2d_transpose(h, weight,OUTPUT_SHAPES[scope],strides=[1,1,1,1],padding="SAME")+bias,
                                 activation_type=ACTIVATION)
                    print('\t\t\tOutput shape: ', h.get_shape().as_list())

            if 'pool' in scope:
                print('\t\tMax Pooling for %s', scope)
                h = tf.nn.max_pool(h,[1,3,3,1],POOL_STRIDES[scope],padding='SAME')
                print('\t\t\tOutput shape: ',h.get_shape().as_list())

    return h


def get_reconstruction_inference(tf_inputs,OUTPUT_SHAPES):
    global TF_WEIGHTS_SCOPE,TF_BIAS_SCOPE,TF_DECONV_SCOPE,POOL_STRIDES
    # forward pass
    for si, scope in enumerate(CONV_SCOPES):
        with tf.variable_scope(TF_RECONSTRUCTION_SCOPE, reuse=True):
            with tf.variable_scope(scope, reuse=True):
                if scope.startswith('conv'):
                    weight, bias = tf.get_variable(TF_WEIGHTS_SCOPE), tf.get_variable(TF_BIAS_SCOPE)
                    print('\t\tConvolution with ReLU activation for ', scope)
                    if si == 0:
                        print('\t\t\tInput shape ', tf_inputs.get_shape().as_list())
                        h = activate(
                            tf.nn.conv2d(tf_inputs, weight, strides=[1,1,1,1], padding='SAME') + bias,
                            activation_type=ACTIVATION, name='hidden')
                        print('\t\t\tOutput shape: ', h.get_shape().as_list())
                    else:
                        h = activate(tf.nn.conv2d(h, weight, strides=[1,1,1,1], padding='SAME') + bias,
                                     activation_type=ACTIVATION, name='hidden')
                        print('\t\t\tOutput shape: ', h.get_shape().as_list())

                if scope.startswith('deconv'):
                    weight, bias = tf.get_variable(TF_WEIGHTS_SCOPE), tf.get_variable(TF_BIAS_SCOPE)

                    if si== len(CONV_SCOPES)-1:
                        print('\t\tConvolution with TanH activation for ', scope)
                        h = tf.nn.tanh(tf.nn.conv2d_transpose(h, weight,OUTPUT_SHAPES[scope],strides=[1,1,1,1],padding="SAME") + bias)
                        print('\t\t\tOutput shape: ', h.get_shape().as_list())

                    else:
                        print('\t\tConvolution with ReLU activation for ', scope)
                        h = activate(tf.nn.conv2d_transpose(h, weight,OUTPUT_SHAPES[scope],strides=[1,1,1,1],padding="SAME")+bias,
                                     activation_type=ACTIVATION)
                        print('\t\t\tOutput shape: ', h.get_shape().as_list())
    return h


def get_prediction(tf_input):
    tf_out = get_inference(tf_input,TEST_OUTPUT_SHAPES)
    tf_rec_out = get_reconstruction_inference(tf_out,TEST_OUTPUT_SHAPES)
    return tf_rec_out


def get_predictions_part(tf_input):
    tf_out = get_inference(tf_input,TEST_OUTPUT_SHAPES)
    return tf_out


def calculate_loss(tf_inputs,tf_labels):
    method = 'weighted'
    l2_decay = True

    if method == 'weighted':
        tf_out = get_inference(tf_inputs, OUTPUT_SHAPES)

        tf_pos_mask = tf.cast(tf.equal(tf_labels, 1), dtype=tf.float32)
        rand_mask = tf.cast(
            tf.greater(tf.truncated_normal(tf_labels.get_shape().as_list(), dtype=tf.float32), 1.1), dtype=tf.float32)
        tf_neg_mask = tf.cast(tf.equal(tf_labels, -1), dtype=tf.float32) * rand_mask
        tf_neut_mask = tf.cast(tf.equal(tf_labels, 0), dtype=tf.float32) * rand_mask

        tf_neut_mask_weighted = tf.cast(tf.equal(tf_labels, 0), dtype=tf.float32) * rand_mask
        tf_pos_mask_weighted = tf.cast(tf.equal(tf_labels, 1), dtype=tf.float32) * 0.99
        tf_neg_mask_weighted = tf.cast(tf.equal(tf_labels, -1), dtype=tf.float32) * 0.99 * rand_mask

        tf_outer_pos_mask = tf.cast(tf.equal(tf_labels, 1), dtype=tf.float32)*9.0 + \
                            tf.cast(tf.equal(tf_labels, -1), dtype=tf.float32) + tf.cast(tf.equal(tf_labels, 0), dtype=tf.float32)

        tf_out_mask_for_logits = tf_pos_mask + tf_neg_mask + tf_neut_mask

        loss = tf.reduce_mean(
            tf.reduce_sum(
                ((tf_out * tf_out_mask_for_logits -
                   tf_labels*(tf_pos_mask_weighted+tf_neg_mask_weighted+tf_neut_mask_weighted)) ** 2) *
                tf_outer_pos_mask,
                axis=[1, 2, 3])) #TODO: check - over all batches?

        if l2_decay:
            for si,scope in enumerate(CONV_SCOPES):
                with tf.variable_scope(scope,reuse=True):
                    loss += beta * tf.reduce_sum(tf.get_variable(TF_WEIGHTS_SCOPE)**2)

    return loss

def calculate_loss_v2(tf_inputs,tf_labels):
    method = 'weighted'
    l2_decay = True

    if method == 'weighted':
        tf_out = get_inference(tf_inputs, OUTPUT_SHAPES)

        tf_pos_mask = tf.cast(tf.equal(tf_labels, 1), dtype=tf.float32)
        tf_neg_mask = tf.cast(tf.equal(tf_labels, -1), dtype=tf.float32)
        tf_neut_mask = tf.cast(tf.equal(tf_labels, 0), dtype=tf.float32)

        pos_importance = (1.0 - tf.reduce_sum(tf_pos_mask))
        neg_importance = (1.0 - tf.reduce_sum(tf_neg_mask))
        neut_importance = (1.0 - tf.reduce_sum(tf_neut_mask))*0.0

        pos_importance /= (pos_importance+neg_importance+neut_importance)
        neg_importance /= (pos_importance + neg_importance + neut_importance)
        neut_importance /= (pos_importance + neg_importance + neut_importance)

        tf_mask = tf_pos_mask*pos_importance + tf_neg_mask*neg_importance + tf_neut_mask*neut_importance

        loss = tf.reduce_mean(
            tf.reduce_sum(
                ((tf_out - tf_labels) ** 2) * tf_mask,
                axis=[1, 2, 3]))

        if l2_decay:
            for si, scope in enumerate(CONV_SCOPES):
                with tf.variable_scope(scope, reuse=True):
                    loss += beta * tf.reduce_sum(tf.get_variable(TF_WEIGHTS_SCOPE) ** 2)

    return loss


def calculate_reconstruction_loss(tf_inputs,tf_labels):
    l2_decay = False

    tf_out = get_inference(tf_inputs, OUTPUT_SHAPES)
    tf_rec_out = get_reconstruction_inference(tf_out, OUTPUT_SHAPES)

    tf_reconstruct_mask = tf.cast(tf.greater(tf_labels, 5), dtype=tf.float32)

    tf_modef_out = tf_rec_out * tf_reconstruct_mask
    tf_modf_labels = tf_labels * tf_reconstruct_mask

    recontruction_loss = tf.reduce_mean(tf.reduce_sum((tf_modef_out-tf_modf_labels)**2,axis=[1,2,3])) + calculate_loss(tf_inputs,tf_labels)

    if l2_decay:
        with tf.variable_scope(TF_RECONSTRUCTION_SCOPE):
            for si, scope in enumerate(CONV_SCOPES):
                with tf.variable_scope(scope, reuse=True):
                    recontruction_loss += beta * tf.reduce_sum(tf.get_variable(TF_WEIGHTS_SCOPE) ** 2)

    return recontruction_loss


def calculate_reconstruction_loss_v2(tf_inputs,tf_labels,tf_rand_offset_height, tf_rand_offset_width):
    l2_decay = True

    tf_out = get_inference(tf_inputs, OUTPUT_SHAPES)

    tf_occlu_mask = tf.ones(shape=[batch_size,10,20,1], dtype=tf.float32)

    tf_occlu_mask_pad = tf.pad(tf_occlu_mask,
                           [[0,0],[tf_rand_offset_height, 30 - (tf_rand_offset_height+10)],
                            [tf_rand_offset_width, 60 - (tf_rand_offset_width+20)], [0,0]])

    #tf_occlu_mask_pad = tf.pad(tf_occlu_mask,[[0,0],[10,10],[20,20],[0,0]])
    tf_occlu_mask_pad = tf.cast(tf.equal(tf_occlu_mask_pad,0.0),dtype=tf.float32)
    tf_rec_out = get_reconstruction_inference(tf_out*tf_occlu_mask_pad, OUTPUT_SHAPES)

    recontruction_loss = tf.reduce_mean(tf.reduce_sum((tf_rec_out-tf_out)**2,axis=[1,2,3])) + \
                         calculate_loss_v2(tf_inputs,tf_labels)

    if l2_decay:
        with tf.variable_scope(TF_RECONSTRUCTION_SCOPE):
            for si, scope in enumerate(CONV_SCOPES):
                with tf.variable_scope(scope, reuse=True):
                    recontruction_loss += beta * tf.reduce_sum(tf.get_variable(TF_WEIGHTS_SCOPE) ** 2)

    return recontruction_loss


def optimize_model(loss,global_step):
    learning_rate = tf.cond(global_step<20,lambda: tf.minimum(tf.train.exponential_decay(0.00001,global_step,2*(file_count//batch_size),1.05,staircase=True),1e-4),
                            lambda: tf.maximum(tf.train.exponential_decay(0.00001,global_step,2*(file_count//batch_size),0.95,staircase=True),1e-7))
    if global_step is not None:
        optimize = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss,global_step)
    else:
        optimize = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss)
    #optimize = tf.train.GradientDescentOptimizer(learning_rate=0.000001).minimize(loss)
    return optimize,learning_rate


def optimize_model_auto_lr(loss,global_step,var_list):
    learning_rate = tf.maximum(tf.train.exponential_decay(0.001,global_step,1,0.9,staircase=True),1e-7)
    if var_list is None:
        optimize = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss)
    else:
        optimize = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss,var_list=var_list)
    #optimize = tf.train.GradientDescentOptimizer(learning_rate=0.000001).minimize(loss)
    return optimize,learning_rate


def inc_gstep(global_step):
    return tf.assign(global_step,global_step+1)

if __name__ == '__main__':
    global sess,graph

    file_count = 500
    data_folder = 'data_reconstruction/'

    width = 60
    height = 30

    channels = 2

    accuracy_low_count = 0
    min_loss = 100000.00

    with sess.as_default() and graph.as_default():
        tf_inpts = tf.placeholder(dtype=tf.float32, shape=[batch_size, height, width, channels], name='inputs')
        tf_labls = tf.placeholder(dtype=tf.float32, shape=[batch_size, height, width, 1], name='labels')
        tf_offset_height = tf.placeholder(dtype=tf.int32, shape=None, name='offset_height')
        tf_offset_width = tf.placeholder(dtype=tf.int32, shape=None, name='offset_width')

        global_step = tf.Variable(0,dtype=tf.int32,trainable=False)
        tf_test_inputs = tf.placeholder(dtype=tf.float32, shape=[1, height, width, channels], name='inputs')
        build_tensorflw_variables()
        build_tensorflw_reconstruction_variables()
        tf_loss = calculate_loss(tf_inpts, tf_labls)
        tf_reconstruction_loss = calculate_reconstruction_loss_v2(tf_inpts, tf_labls,tf_offset_height, tf_offset_width)

        tf_prediction = get_prediction(tf_test_inputs)
        tf_part_prediction = get_predictions_part(tf_test_inputs)

        tf_optimize, tf_learning_rate = optimize_model_auto_lr(tf_loss, global_step,None)
        var_list = []
        for v in tf.global_variables():
            if v.name.startswith(TF_RECONSTRUCTION_SCOPE):
                var_list.append(v)
                print(v.name)
        tf_rec_optimize, _ = optimize_model_auto_lr(tf_reconstruction_loss, global_step,var_list)
        tf_inc_gstep = inc_gstep(global_step)

        tf.global_variables_initializer().run()
        for epoch in range(250):
            avg_loss = []
            avg_loss_rec = []
            for step in range(file_count//batch_size):
                inp1, lbl1 = load_data.load_batch_npz(data_folder, batch_size, height, width, channels, file_count,
                                                      shuffle=True, rand_cover_percentage=None, flip_lr=False, flip_ud= False)


                #assert np.max(lbl1)==1.0 and np.min(lbl1)==-1.0, '%.5f, %.5f'%(np.max(lbl1),np.min(lbl1))

                #print('occupied ratio: ',occupied_size/lbl1.size)
                l, labels, _ = sess.run([tf_loss, tf_labls, tf_optimize], feed_dict={tf_inpts:inp1/normalize_constant, tf_labls:lbl1})
                #l, labels = sess.run([tf_loss, tf_labls],
                #                        feed_dict={tf_inpts: inp1 / normalize_constant, tf_labls: lbl1})
                rec_l, _ = sess.run([tf_reconstruction_loss,tf_rec_optimize], feed_dict={tf_inpts:inp1/normalize_constant, tf_labls:lbl1,
                                                                                         tf_offset_height:np.random.randint(0,20),
                                                                                         tf_offset_width:np.random.randint(0,40)})
                avg_loss.append(l)
                avg_loss_rec.append(rec_l)

            if (epoch+1)%1==0:

                xx, yy = np.meshgrid(np.arange(-299, 300, 2), np.arange(1, 300, 2))
                test_full_map = -100 + np.zeros(xx.shape)
                test_full_map_part = -100 + np.zeros(xx.shape)

                for col_no in range(0, 150, 30):
                    for row_no in range(0, 300, 60):
                        local_xx = xx[col_no:col_no + 30, row_no:row_no + 60][:, :, np.newaxis]
                        local_yy = yy[col_no:col_no + 30, row_no:row_no + 60][:, :, np.newaxis]
                        test_input = np.concatenate((local_xx, local_yy), axis=2)[np.newaxis, :, :, :]

                        pred = sess.run(tf_prediction, feed_dict={tf_test_inputs:test_input/normalize_constant})
                        pred_part = sess.run(tf_part_prediction, feed_dict = {tf_test_inputs:test_input/normalize_constant})
                        test_full_map[col_no:col_no + 30, row_no:row_no + 60] = pred[0, :, :,0]
                        test_full_map_part[col_no:col_no + 30, row_no:row_no + 60] = pred_part[0, :, :, 0]

                test_full_map = ndimage.gaussian_filter(test_full_map, sigma=(2.5, 2.5))

                plt.close('all')
                plt.figure(figsize=(12, 10))
                plt.subplot(221)
                plt.title('Occupancy $\in [elastic, elastic]$')
                plt.scatter(xx.ravel(), yy.ravel(), c=test_full_map.ravel(), s=1, cmap='jet')
                plt.colorbar()
                plt.xlim([-300, 300]); plt.ylim([0, 300])
                plt.axis('equal')
                plt.subplot(223)
                plt.title('Occupancy $\in [-1, 1]$')
                plt.scatter(xx.ravel(), yy.ravel(), c=test_full_map.ravel(), s=1, cmap='jet', vmin=-1, vmax=1)
                plt.colorbar()
                plt.xlim([-300, 300]); plt.ylim([0, 300])
                plt.axis('equal')

                plt.subplot(222)
                plt.title('Occupancy (Part) $\in [elastic, elastic]$')
                plt.scatter(xx.ravel(), yy.ravel(), c=test_full_map_part.ravel(), s=1, cmap='jet')
                plt.colorbar()
                plt.xlim([-300, 300]);
                plt.ylim([0, 300])
                plt.axis('equal')
                plt.subplot(224)
                plt.title('Occupancy (Part) $\in [-1, 1]$')
                plt.scatter(xx.ravel(), yy.ravel(), c=test_full_map_part.ravel(), s=1, cmap='jet', vmin=-1, vmax=1)
                plt.colorbar()
                plt.xlim([-300, 300]);
                plt.ylim([0, 300])
                plt.axis('equal')
                #plt.show()
                plt.savefig(pred_data_dir+os.sep+'pred_%d.png'%(epoch+1))

            curr_loss = np.mean(avg_loss)

            if curr_loss > min_loss:
                accuracy_low_count += 1
                print('\tIncreasing the accuracy_low_count to ',accuracy_low_count)

            if accuracy_low_count >= ACCURACY_DROP_CAP:
                print('\tAccuracy drop exceeded the threshold')
                sess.run(tf_inc_gstep)
                accuracy_low_count = 0

            print('Loss Epoch (%d): %.5f (Prediction) %.5f(Reconstruction)'%(epoch,np.mean(avg_loss),np.mean(avg_loss_rec)))
            print('\tLearning rate: ',sess.run([tf_learning_rate,global_step]))

            if curr_loss < min_loss:
                min_loss = curr_loss