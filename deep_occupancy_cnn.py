from __future__ import division #python 2 division

import numpy as np
import tensorflow as tf
import load_data
import matplotlib.pyplot  as plt
import os
from skimage import io,img_as_uint
import sys
from scipy import ndimage
import  getopt
import sys

CONV_SCOPES = None
VAR_SHAPES,TRAIN_OUTPUT_SHAPES,TEST_OUTPUT_SHAPES = None,None,None

# in 30,60
# after pool1 : 15,30
# after pool2 : 5,10
conv1kernel = None
conv2kernel = None

batch_size = 10
OUTPUT_TYPE = 'regression'

TF_WEIGHTS_SCOPE = 'weights'
TF_BIAS_SCOPE = 'bias'
TF_DECONV_SCOPE = 'deconv'
TF_MU = 'mu'
TF_SIGMA = 'sigma'
TF_BETA = 'beta'
TF_GAMMA = 'gamma'

ACTIVATION = 'relu'
beta = 0.00001
start_lr = 0.001
AUTO_DECREASE_LR = True
ACCURACY_DROP_CAP = 5

USE_BN = False

mean_var_decay = 0.59

def define_hyperparameters(width,height,exp_type):
    global CONV_SCOPES,VAR_SHAPES,TRAIN_OUTPUT_SHAPES,TEST_OUTPUT_SHAPES,start_lr,batch_size,beta

    CONV_SCOPES = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'deconv3', 'deconv2', 'deconv1','fulcon']

    conv1kernel_x,conv1kernel_y = 6,3
    conv2kernel_x, conv2kernel_y = 6,3

    batch_size = 10
    OUTPUT_TYPE = 'regression'


    if OUTPUT_TYPE == 'regression':
        if exp_type != 'intel':
            VAR_SHAPES = {'conv1': [conv1kernel_y,conv1kernel_x,2,16], 'conv2': [1,1,16,16],
                          'conv3':[conv2kernel_y,conv2kernel_x,16,32], 'conv4': [1,1,32,32],
                          'conv5': [conv2kernel_y, conv2kernel_x, 32, 32], 'conv6': [1, 1, 32, 32],
                          'deconv3':[1,1,32,32],'deconv2':[1,1,16,32],'deconv1':[1,1,1,16],'fulcon':[1,height,width,1]}
            TRAIN_OUTPUT_SHAPES = {'deconv3':[batch_size,height, width,32],'deconv2':[batch_size,height, width,16],'deconv1':[batch_size,height,width,1]}
            TEST_OUTPUT_SHAPES = {'deconv3':[1,height,width,32],'deconv2':[1,height,width,16],'deconv1':[1,height,width,1]}
        else:
            batch_size=25
            beta = 1e-5
            VAR_SHAPES = {'conv1': [conv1kernel_y, conv1kernel_x, 2, 32], 'conv2': [1, 1, 32, 32],
                          'conv3': [conv2kernel_y, conv2kernel_x, 32, 64], 'conv4': [1, 1, 64, 64],
                          'conv5': [conv2kernel_y, conv2kernel_x, 64, 64], 'conv6': [1, 1, 64, 64],
                          'deconv3': [1, 1 , 64, 64], 'deconv2': [1, 1, 32, 64],
                          'deconv1': [1, 1, 1, 32],'fulcon':[1,height,width,1]}
            TRAIN_OUTPUT_SHAPES = {'deconv3': [batch_size, height, width, 64],
                                   'deconv2': [batch_size, height, width, 32],
                                   'deconv1': [batch_size, height, width, 1]}
            TEST_OUTPUT_SHAPES = {'deconv3': [1, height, width, 64], 'deconv2': [1, height, width, 32],
                                  'deconv1': [1, height, width, 1]}

            start_lr = 5e-4

    elif OUTPUT_TYPE == 'classification':
        raise NotImplementedError
    else:
        raise NotImplementedError

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
    elif activation_type=='elu':
        return tf.nn.elu(x)
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

            if scope == 'fulcon':
                print('Build fulcon variable')
                tf.get_variable(TF_WEIGHTS_SCOPE, shape=VAR_SHAPES[scope],
                                initializer=tf.contrib.layers.xavier_initializer())
                print('Build weight')
                tf.get_variable(TF_BIAS_SCOPE,
                                initializer=tf.random_uniform(shape=VAR_SHAPES[scope], minval=-0.01,
                                                              maxval=0.01, dtype=tf.float32))
                print('Build bias')


    print([v.name for v in tf.global_variables()])


def build_bn_variables(tf_inputs):
    global CONV_SCOPES,TRAIN_OUTPUT_SHAPES

    for si,scope in enumerate(CONV_SCOPES):
            if scope.startswith('conv'):
                with tf.variable_scope(scope, reuse=True):
                    weight, bias = tf.get_variable(TF_WEIGHTS_SCOPE), tf.get_variable(TF_BIAS_SCOPE)
                print('\t\tConvolution with ReLU activation for ', scope)
                if si == 0:
                    print('\t\t\tInput shape ', tf_inputs.get_shape().as_list())
                    h = activate(
                        tf.nn.conv2d(tf_inputs, weight, strides=[1,1,1,1], padding='VALID') + bias,
                        activation_type=ACTIVATION, name='hidden')
                    print('\t\t\tOutput shape: ', h.get_shape().as_list())
                else:
                    h = activate(tf.nn.conv2d(h, weight, strides=[1,1,1,1], padding='VALID') + bias,
                                 activation_type=ACTIVATION, name='hidden')
                    print('\t\t\tOutput shape: ', h.get_shape().as_list())

                h_shape = h.get_shape().as_list()

                with tf.variable_scope(scope, reuse=False):
                    tf.get_variable(TF_MU, initializer = tf.zeros(shape=[1]+h_shape[1:],dtype=tf.float32),trainable=False)
                    tf.get_variable(TF_SIGMA, initializer=tf.zeros(shape=[1]+h_shape[1:], dtype=tf.float32),trainable=False)
                    tf.get_variable(TF_BETA, initializer=tf.zeros(shape=[1]+h_shape[1:], dtype=tf.float32))
                    tf.get_variable(TF_GAMMA, initializer=tf.random_uniform(minval=0.9,maxval=1.0,shape=[1]+h_shape[1:], dtype=tf.float32))

                height_fill = (height - h_shape[1])
                width_fill = (width - h_shape[2])
                print(h_shape, ' ', height_fill, ' ', width_fill)
                h = tf.pad(h, [[0, 0], [0, height_fill], [0, width_fill], [0, 0]], mode='SYMMETRIC')
                h_shape = h.get_shape().as_list()

                assert h_shape[1] == height and h_shape[2] == width

            if scope.startswith('deconv'):
                with tf.variable_scope(scope, reuse=True):
                    weight, bias = tf.get_variable(TF_WEIGHTS_SCOPE), tf.get_variable(TF_BIAS_SCOPE)

                if si == len(CONV_SCOPES)-1:
                    if OUTPUT_TYPE == 'regression':
                        print('\t\tConvolution with TanH activation for ', scope)
                        h = tf.nn.tanh(tf.nn.conv2d_transpose(h, weight, TRAIN_OUTPUT_SHAPES[scope],strides=[1,1,1,1],padding="SAME") + bias)
                        print('\t\t\tOutput shape: ', h.get_shape().as_list())

                    elif OUTPUT_TYPE == 'classification':
                        print('\t\tConvolution with logits for ', scope)
                        h = tf.nn.conv2d_transpose(h, weight, TRAIN_OUTPUT_SHAPES[scope],
                                                   strides=[1, 1, 1, 1], padding="SAME") + bias
                        print('\t\t\tOutput shape: ', h.get_shape().as_list())

                else:
                    print('\t\tConvolution with ReLU activation for ', scope)
                    h = activate(tf.nn.conv2d_transpose(h, weight,TRAIN_OUTPUT_SHAPES[scope],strides=[1,1,1,1],padding="SAME")+bias,
                                 activation_type=ACTIVATION)
                    print('\t\t\tOutput shape: ', h.get_shape().as_list())

                    h_shape = h.get_shape().as_list()
                    with tf.variable_scope(scope, reuse=False):
                        tf.get_variable(TF_MU, initializer=tf.zeros(shape=[1]+h_shape[1:], dtype=tf.float32), trainable=False)
                        tf.get_variable(TF_SIGMA, initializer=tf.zeros(shape=[1]+h_shape[1:], dtype=tf.float32), trainable=False)
                        tf.get_variable(TF_BETA, initializer=tf.zeros(shape=[1]+h_shape[1:], dtype=tf.float32))
                        tf.get_variable(TF_GAMMA, initializer=tf.random_uniform(minval=0.9,maxval=1.0,shape=[1]+h_shape[1:], dtype=tf.float32))


def get_inference(tf_inputs,OUTPUT_SHAPES,is_training):
    global TF_WEIGHTS_SCOPE,TF_BIAS_SCOPE,TF_DECONV_SCOPE,POOL_STRIDES, USE_BN
    # forward pass

    for si, scope in enumerate(CONV_SCOPES):
        with tf.variable_scope(scope, reuse=True) as sc:

            if scope.startswith('conv'):
                weight, bias = tf.get_variable(TF_WEIGHTS_SCOPE), tf.get_variable(TF_BIAS_SCOPE)
                print('\t\tConvolution with ReLU activation for ', scope)
                if si == 0:
                    print('\t\t\tInput shape ', tf_inputs.get_shape().as_list())
                    if not USE_BN:
                        h = tf.nn.conv2d(tf_inputs, weight, strides=[1,1,1,1], padding='VALID') + bias
                    else:
                        h = tf.nn.conv2d(tf_inputs, weight, strides=[1, 1, 1, 1], padding='VALID')
                    print('\t\t\tOutput shape: ', h.get_shape().as_list())
                else:
                    if not USE_BN:
                        h = tf.nn.conv2d(h, weight, strides=[1,1,1,1], padding='VALID') + bias
                    else:
                        h = tf.nn.conv2d(h, weight, strides=[1, 1, 1, 1], padding='VALID')
                    print('\t\t\tOutput shape: ', h.get_shape().as_list())

                if USE_BN:
                    if is_training:
                        mean, var = tf.nn.moments(h,axes=[0],keep_dims=True)
                        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(tf.get_variable(TF_MU),mean_var_decay * tf.get_variable(TF_MU) + (1-mean_var_decay)* mean))
                        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(tf.get_variable(TF_SIGMA), mean_var_decay * tf.get_variable(TF_SIGMA) + (1-mean_var_decay)* var))

                        h = tf.nn.batch_normalization(h,mean,var,tf.get_variable(TF_BETA),tf.get_variable(TF_GAMMA),1e-6)
                    else:
                        h = tf.nn.batch_normalization(h, tf.get_variable(TF_MU), tf.get_variable(TF_SIGMA),None,None, 1e-6)

                h = activate(h,activation_type=ACTIVATION)

                h_shape = h.get_shape().as_list()
                assert h_shape[0]==tf_inputs.get_shape().as_list()[0]

                height_fill = (height - h_shape[1])
                width_fill = (width - h_shape[2])
                print(h_shape, ' ', height_fill, ' ', width_fill)
                h = tf.pad(h,[[0,0],[0,height_fill],[0,width_fill],[0,0]],mode='SYMMETRIC')
                h_shape = h.get_shape().as_list()

                assert h_shape[1]==height and h_shape[2]==width

            if scope.startswith('deconv'):
                weight, bias = tf.get_variable(TF_WEIGHTS_SCOPE), tf.get_variable(TF_BIAS_SCOPE)


                print('\t\tConvolution with ReLU activation for ', scope)
                if not USE_BN:
                    h = tf.nn.conv2d_transpose(h, weight,OUTPUT_SHAPES[scope],strides=[1,1,1,1],padding="SAME")+bias
                else:
                    h = tf.nn.conv2d_transpose(h, weight, OUTPUT_SHAPES[scope], strides=[1, 1, 1, 1],
                                               padding="SAME")
                print('\t\t\tOutput shape: ', h.get_shape().as_list())

                if USE_BN:
                    if is_training:
                        mean, var = tf.nn.moments(h, axes=[0], keep_dims=True)
                        tf.add_to_collection(
                            tf.GraphKeys.UPDATE_OPS,
                            tf.assign(tf.get_variable(TF_MU),
                                      mean_var_decay * tf.get_variable(TF_MU) + (1 - mean_var_decay) * mean)
                        )
                        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(tf.get_variable(TF_SIGMA),
                                                        mean_var_decay * tf.get_variable(TF_SIGMA) + (
                                                        1 - mean_var_decay) * var))

                        h = tf.nn.batch_normalization(h, mean, var,
                                                      tf.get_variable(TF_BETA), tf.get_variable(TF_GAMMA),1e-6)
                    else:
                        h = tf.nn.batch_normalization(h, tf.get_variable(TF_MU), tf.get_variable(TF_SIGMA), None, None,
                                                      1e-6)

                h = activate(h, activation_type=ACTIVATION)
            if scope=='fulcon':
                weight, bias = tf.get_variable(TF_WEIGHTS_SCOPE), tf.get_variable(TF_BIAS_SCOPE)

                h = tf.multiply(weight,h) + bias
                h = tf.nn.tanh(h)

            if 'pool' in scope:
                raise NotImplementedError

    return h


def get_prediction(tf_input):
    print('Defining inference function for Predictions')
    tf_out = get_inference(tf_input,TEST_OUTPUT_SHAPES,False)
    return tf_out


def get_predictions_with_ohe(tf_input):
    raise NotImplementedError

def get_predictions_with_croppin_and_padding(tf_input):
    tf_out = get_inference(tf_input, TEST_OUTPUT_SHAPES,False)
    tf_crop_out = tf.map_fn(lambda x: tf.image.resize_image_with_crop_or_pad(x,26,56),tf_out)
    tf_pad_out = tf.pad(tf_crop_out,[[0,0],[2,2],[2,2],[0,0]])
    return tf_pad_out


def calculate_loss_one_hot(tf_inputs,tf_labels):

    raise NotImplementedError

def calculate_loss(tf_inputs,tf_labels):
    method = 'weighted'
    l2_decay = True

    if method == 'naive':
        tf_out = get_inference(tf_inputs,TRAIN_OUTPUT_SHAPES)

        tf_out_pos_mask = tf.cast(tf.equal(tf_labels,1),dtype=tf.float32)
        tf_out_neg_mask = tf.cast(tf.equal(tf_labels,-1),dtype=tf.float32)
        tf_out_mask = tf_out_pos_mask + tf_out_neg_mask

        loss = tf.reduce_mean(tf.reduce_sum(((tf_out-tf_labels)**2)*tf_out_mask,axis=[1,2,3]))

    elif method == 'weighted':
        tf_out = get_inference(tf_inputs, TRAIN_OUTPUT_SHAPES)

        tf_pos_mask = tf.cast(tf.equal(tf_labels, 1), dtype=tf.float32)

        rand_mask = tf.cast(
            tf.greater(tf.truncated_normal(tf_labels.get_shape().as_list(), dtype=tf.float32), 1.1), dtype=tf.float32)
        tf_neg_mask = tf.cast(tf.equal(tf_labels, -1), dtype=tf.float32) * rand_mask

        tf_pos_mask_weighted = tf.cast(tf.equal(tf_labels, 1), dtype=tf.float32) * 0.95
        tf_neg_mask_weighted = tf.cast(tf.equal(tf_labels, -1), dtype=tf.float32) * 0.95 * rand_mask

        tf_outer_pos_mask = tf.cast(tf.equal(tf_labels, 1), dtype=tf.float32)*10.0 + \
                            tf.cast(tf.equal(tf_labels, -1), dtype=tf.float32)

        tf_out_mask_for_logits = tf_pos_mask + tf_neg_mask

        loss = tf.reduce_mean(
            tf.reduce_sum(
                ((tf_out * tf_out_mask_for_logits -
                   tf_labels*(tf_pos_mask_weighted+tf_neg_mask_weighted)) ** 2) *
                tf_outer_pos_mask,
                axis=[1, 2, 3])) #TODO: check - over all batches?

        if l2_decay:
            for si,scope in enumerate(CONV_SCOPES):
                with tf.variable_scope(scope,reuse=True):
                    loss += beta * tf.reduce_sum(tf.get_variable(TF_WEIGHTS_SCOPE)**2)

    elif method == 'equal_number_of_posneg_samples':
        tf_out = get_inference(tf_inputs, TRAIN_OUTPUT_SHAPES)

        tf_out_pos_mask = tf.cast(tf.equal(tf_labels, 1), dtype=tf.float32)
        tf_out_neg_mask = tf.cast(tf.equal(tf_labels, -1), dtype=tf.float32)
        tf_out_neg_rand_mask = tf.cast(tf.greater(tf.truncated_normal(tf_labels.get_shape().as_list(),dtype=tf.float32),0.0),dtype=tf.float32)
        tf_out_neg_mask = tf_out_neg_mask*tf_out_neg_rand_mask
        tf_out_mask = tf_out_pos_mask + tf_out_neg_mask
        loss = tf.reduce_mean(tf.reduce_sum(((tf_out - tf_labels) ** 2) , axis=[1, 2, 3]))

    return loss

def calculate_loss_v2(tf_out,tf_labels):

    l2_decay = True

    tf_pos_mask = tf.cast(tf.equal(tf_labels, 1), dtype=tf.float32)
    tf_neg_mask = tf.cast(tf.equal(tf_labels, -1), dtype=tf.float32)
    tf_neut_mask = tf.cast(tf.equal(tf_labels, 0), dtype=tf.float32)

    pos_importance = tf.reduce_sum(tf_neg_mask) + tf.reduce_sum(tf_neut_mask)
    neg_importance = tf.reduce_sum(tf_pos_mask) + tf.reduce_sum(tf_neut_mask)
    neut_importance = tf.reduce_sum(tf_neg_mask) + tf.reduce_sum(tf_pos_mask)

    pos_importance /= (pos_importance + neg_importance + neut_importance)
    neg_importance /= (pos_importance + neg_importance + neut_importance)
    neut_importance /= (pos_importance + neg_importance + neut_importance)

    if exp_type != 'intel':
        tf_mask = tf_pos_mask * (5.0*(1.0+pos_importance))**2 + tf_neg_mask * (1.0+neg_importance)**2 + \
                  tf_neut_mask * (1.0 + neut_importance)**2
    else:
        tf_mask = tf_pos_mask * (5.0*(1.0 + pos_importance))**2 + tf_neg_mask * (2.5*(1.0 + neg_importance))**2 + \
                  tf_neut_mask * (1.0 + neut_importance)**2
    loss = tf.reduce_mean(
        tf.reduce_sum(
            ((tf_out - tf_labels) ** 2) * tf_mask ,
            axis=[1, 2, 3]))

    if l2_decay:
        for si, scope in enumerate(CONV_SCOPES):
            with tf.variable_scope(scope, reuse=True):
                loss += beta * tf.reduce_sum(tf.get_variable(TF_WEIGHTS_SCOPE) ** 2)

    return loss

def optimize_model(loss,global_step):
    learning_rate = tf.cond(global_step<20,lambda: tf.minimum(tf.train.exponential_decay(0.00001,global_step,2*(file_count//batch_size),1.05,staircase=True),1e-4),
                            lambda: tf.maximum(tf.train.exponential_decay(0.00001,global_step,2*(file_count//batch_size),0.95,staircase=True),1e-7))
    optimize = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss,global_step)
    #optimize = tf.train.GradientDescentOptimizer(learning_rate=0.000001).minimize(loss)
    return optimize,learning_rate


def optimize_model_auto_lr(loss,global_step):
    # LEARNING RATE before changing 0.00001
    lr = start_lr if not USE_BN else 0.00001
    learning_rate = tf.maximum(tf.train.exponential_decay(lr,global_step,1,0.9,staircase=True),1e-7)
    optimize = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss)
    #optimize = tf.train.GradientDescentOptimizer(learning_rate=0.000001).minimize(loss)
    return optimize,learning_rate


def inc_gstep(global_step):

    return tf.assign(global_step,global_step+1)

test_width = 600
test_height = 300
pred_data_dir = 'pred_data_special_padding'

normalize_constant = 150

if __name__ == '__main__':

    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "", ["experiment-type=", "output-dir="])
    except getopt.GetoptError as err:
        print(err.with_traceback())

    if len(opts) != 0:
        for opt, arg in opts:
            if opt == '--output-dir':
                pred_data_dir = arg
            if opt == '--experiment-type':
                exp_type = arg

    if exp_type=='static':
        data_folder = 'data/'
        file_count = 1300
        width = 60
        height = 30
    elif exp_type=='dynamic':
        data_folder = 'data_dynamic/'
        file_count = 335
        width = 100  # 60
        height = 50  # 30
    elif exp_type=='intel':
        data_folder = 'data_intel/'
        file_count = 908
        width = 140
        height = 70
        normalize_constant = 7.0

    if not os.path.exists(pred_data_dir):
        os.mkdir(pred_data_dir)


    #file_count = 1300 (data/) file_count = 335 (data_dynamic/)

    channels = 2

    accuracy_low_count = 0
    min_loss = 100000.00


    define_hyperparameters(width,height,exp_type)

    with sess.as_default() and graph.as_default():
        tf_inpts = tf.placeholder(dtype=tf.float32, shape=[batch_size, height, width, channels], name='inputs')
        tf_labls = tf.placeholder(dtype=tf.float32, shape=[batch_size, height, width, 1], name='labels')
        global_step = tf.Variable(0,dtype=tf.int32,trainable=False)
        tf_test_inputs = tf.placeholder(dtype=tf.float32, shape=[1, height, width, channels], name='inputs')
        build_tensorflw_variables()
        if USE_BN:
            build_bn_variables(tf_inpts)

        if OUTPUT_TYPE=='classification':
            raise NotImplementedError
        elif OUTPUT_TYPE=='regression':
            tf_out = get_inference(tf_inpts, TRAIN_OUTPUT_SHAPES, True)
            tf_loss = calculate_loss_v2(tf_out, tf_labls)
            tf_prediction = get_prediction(tf_test_inputs)
        else:
            raise NotImplementedError

        if not AUTO_DECREASE_LR:
            tf_optimize, tf_learning_rate = optimize_model(tf_loss,global_step)
        else:
            tf_optimize, tf_learning_rate = optimize_model_auto_lr(tf_loss, global_step)
            tf_inc_gstep = inc_gstep(global_step)

        tf.global_variables_initializer().run()
        for epoch in range(500):
            avg_loss = []
            for step in range(file_count//batch_size):
                inp1, lbl1 = load_data.load_batch_npz(data_folder, batch_size, height, width, channels, file_count,
                                                      shuffle=True,rand_cover_percentage=None, flip_lr=True, flip_ud=True,exp_type=exp_type)

                occupied_size  = np.where(lbl1==1)[0].size
                #print('occupied ratio: ',occupied_size/lbl1.size)
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    l, labels, _ = sess.run([tf_loss, tf_labls, tf_optimize], feed_dict={tf_inpts:inp1/normalize_constant, tf_labls:lbl1})
                    avg_loss.append(l)

            if (epoch+1)%1==0:

                if exp_type=='static':
                    xx, yy = np.meshgrid(np.arange(-299, 300, 2), np.arange(1, 300, 2))
                    test_full_map = -100 + np.zeros(xx.shape)

                    for col_no in range(0, 150, height):
                        for row_no in range(0, 300, width):
                            local_xx = xx[col_no:col_no + height, row_no:row_no + width][:, :, np.newaxis]
                            local_yy = yy[col_no:col_no + height, row_no:row_no + width][:, :, np.newaxis]
                            test_input = np.concatenate((local_xx, local_yy), axis=2)[np.newaxis, :, :, :]

                            pred = sess.run(tf_prediction, feed_dict={tf_test_inputs:test_input/normalize_constant})

                            test_full_map[col_no:col_no + height, row_no:row_no + width] = pred[0, :, :,0]

                    test_full_map = ndimage.gaussian_filter(test_full_map, sigma=(1.5, 1.5))

                if exp_type=='dynamic':
                    xx, yy = np.meshgrid(np.arange(-100, 100, 1), np.arange(0, 100, 1))
                    test_full_map = -100 + np.zeros(xx.shape)

                    for col_no in range(0, 100, height):
                        for row_no in range(0, 200, width):
                            local_xx = xx[col_no:col_no + height, row_no:row_no + width][:, :, np.newaxis]
                            local_yy = yy[col_no:col_no + height, row_no:row_no + width][:, :, np.newaxis]
                            test_input = np.concatenate((local_xx, local_yy), axis=2)[np.newaxis, :, :, :]

                            pred = sess.run(tf_prediction, feed_dict={tf_test_inputs: test_input / normalize_constant})

                            test_full_map[col_no:col_no + height, row_no:row_no + width] = pred[0, :, :, 0]

                if exp_type == 'intel':
                    xx, yy = np.meshgrid(np.arange(-35, 35, 0.1), np.arange(-25, 10, 0.1))
                    test_full_map = -100 + np.zeros(xx.shape)

                    for col_no in range(0,350,height):
                        for row_no in range(0,700,width):
                            local_xx = xx[col_no:col_no + height, row_no:row_no + width][:, :, np.newaxis]
                            local_yy = yy[col_no:col_no + height, row_no:row_no + width][:, :, np.newaxis]
                            test_input = np.concatenate((local_xx, local_yy), axis=2)[np.newaxis, :, :, :]

                            pred = sess.run(tf_prediction,
                                            feed_dict={tf_test_inputs: test_input / normalize_constant})
                            test_full_map[col_no:col_no + height, row_no:row_no + width] = pred[0, :, :, 0]

                    #test_full_map = ndimage.gaussian_filter(test_full_map, sigma=(1.5, 1.5))
                # Smoothing test_full_map

                '''for col_no in range(0, 120, 30):
                    for row_no in range(0, 240, 60):
                        local_xx = xx[(col_no+15):(col_no+15) + 30, (row_no+30):(row_no+30) + 60][:, :, np.newaxis]
                        local_yy = yy[(col_no+15):(col_no+15) + 30, (row_no+30):(row_no+30) + 60][:, :, np.newaxis]

                        test_input = np.concatenate((local_xx, local_yy), axis=2)[np.newaxis, :, :, :]
                        pred = sess.run(tf_prediction, feed_dict={tf_test_inputs:test_input/normalize_constant})

                        test_full_map[(col_no+25):(col_no+25) + 10, (row_no+40):(row_no+40) + 40] = pred[0, 10:20, 10:50,0]'''

                plt.close('all')
                plt.figure(figsize=(12, 10))
                plt.subplot(211)
                plt.title('Occupancy $\in [elastic, elastic]$')
                #plt.scatter(xx.ravel(), yy.ravel(), c=test_full_map.ravel(), s=1, cmap='jet')
                plt.imshow(test_full_map,cmap='jet',interpolation=None)
                plt.colorbar()
                plt.xlim([-300, 300]); plt.ylim([0, 300])
                plt.axis('equal')
                plt.subplot(212)
                plt.title('Occupancy $\in [-1, 1]$')
                #plt.scatter(xx.ravel(), yy.ravel(), c=test_full_map.ravel(), s=1, cmap='jet', vmin=-1, vmax=1)
                plt.imshow(test_full_map,cmap='jet',interpolation=None, vmin=-1, vmax=1)
                plt.colorbar()
                plt.xlim([-300, 300]); plt.ylim([0, 300])
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

            print('Loss Epoch (%d): %.5f'%(epoch,np.mean(avg_loss)))
            print('\tLearning rate: ',sess.run([tf_learning_rate,global_step]))

            if curr_loss < min_loss:
                min_loss = curr_loss