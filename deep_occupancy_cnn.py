from __future__ import division #python 2 division

import numpy as np
import tensorflow as tf
import load_data
import matplotlib.pyplot  as plt
import os
from skimage import io,img_as_uint
import sys

CONV_SCOPES = ['conv1','conv2','deconv2','deconv1']
# in 30,60
# after pool1 : 15,30
# after pool2 : 5,10
conv1kernel = 2
conv2kernel = 2

batch_size = 2
OUTPUT_TYPE = 'regression'
if OUTPUT_TYPE == 'regression':
    VAR_SHAPES = {'conv1': [conv1kernel,conv1kernel,2,16],'conv2':[conv2kernel,conv2kernel,16,32],'deconv2':[conv2kernel,conv2kernel,16,32],'deconv1':[2,2,1,16]}
    OUTPUT_SHAPES = {'deconv2':[batch_size,30, 60,16],'deconv1':[batch_size,30,60,1]}
    TEST_OUTPUT_SHAPES = {'deconv2':[1,30, 60,16],'deconv1':[1,30,60,1]}
elif OUTPUT_TYPE == 'classification':
    VAR_SHAPES = {'conv1': [3, 3, 2, 32], 'conv2': [3, 3, 32, 64], 'deconv2': [3, 3, 32, 64], 'deconv1': [1, 1, 3, 32]}
    OUTPUT_SHAPES = {'deconv2': [batch_size, 30, 60, 32], 'deconv1': [batch_size, 30, 60, 3]}
    TEST_OUTPUT_SHAPES = {'deconv2': [1, 30, 60, 32], 'deconv1': [1, 30, 60, 3]}
else:
    raise NotImplementedError

POOL_STRIDES = {'pool1':[1,2,2,1],'pool2':[1,3,3,1]}
TF_WEIGHTS_SCOPE = 'weights'
TF_BIAS_SCOPE = 'bias'
TF_DECONV_SCOPE = 'deconv'
ACTIVATION = 'relu'
beta = 0.00001

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
                    tf.get_variable(TF_WEIGHTS_SCOPE,shape=VAR_SHAPES[scope],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
                    tf.get_variable(TF_BIAS_SCOPE, VAR_SHAPES[scope][-1],
                                           initializer = tf.constant_initializer(0.001,dtype=tf.float32))

                if scope.startswith('deconv'):
                    tf.get_variable(TF_WEIGHTS_SCOPE, shape=VAR_SHAPES[scope],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1,
                                                                                          dtype=tf.float32))
                    tf.get_variable(TF_BIAS_SCOPE, VAR_SHAPES[scope][-2],
                                           initializer=tf.constant_initializer(0.001, dtype=tf.float32))
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


def get_prediction(tf_input):
    tf_out = get_inference(tf_input,TEST_OUTPUT_SHAPES)
    return tf_out


def get_predictions_with_ohe(tf_input):
    if OUTPUT_TYPE != 'classification':
        raise Exception
    tf_out = get_inference(tf_input,TEST_OUTPUT_SHAPES)
    tf_out_args = tf.argmax(tf.nn.softmax(tf_out),axis=3)
    tf_out_args = tf.expand_dims(tf_out_args,axis=-1)-1
    return tf_out_args


def calculate_loss_one_hot(tf_inputs,tf_labels):
    tf_labels = tf.cast(tf.squeeze(tf_labels),dtype=tf.int32)
    tf_labels_ohe = tf.one_hot(tf_labels,3,1.0,0.0,axis=-1)

    method = 'weighted'

    if method == 'weighted':
        tf_out = get_inference(tf_inputs, OUTPUT_SHAPES)
        tf_pos_masks, tf_neg_masks = [],[]

        tf_pos_mask = tf.cast(tf.equal(tf_labels, 1), dtype=tf.float32) * 1.1
        tf_neg_mask = tf.cast(tf.equal(tf_labels, -1), dtype=tf.float32) * 0.9  # TODO: 0.1 bias?
        tf_posneg_mask = tf_pos_mask + tf_neg_mask

        tf_labels_shape = tf_labels_ohe.get_shape().as_list()
        print(tf_labels_shape)
        tf_out_reshaped = tf.reshape(tf_out,[tf_labels_shape[0],tf_labels_shape[1]*tf_labels_shape[2],tf_labels_shape[3]])
        tf_posneg_reshaped = tf.reshape(tf_posneg_mask,[tf_labels_shape[0],tf_labels_shape[1]*tf_labels_shape[2],1])
        tf_labels_reshaped = tf.reshape(tf_labels_ohe,[tf_labels_shape[0],tf_labels_shape[1]*tf_labels_shape[2],tf_labels_shape[3]])

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels_reshaped,logits=tf_out_reshaped)*tf.squeeze(tf_posneg_reshaped)
        ) #TODO: check - over all batches?
        return loss
    else:
        raise NotImplementedError


def calculate_loss(tf_inputs,tf_labels):
    method = 'weighted'
    l2_decay = True
    if method == 'naive':
        tf_out = get_inference(tf_inputs,OUTPUT_SHAPES)

        tf_out_pos_mask = tf.cast(tf.equal(tf_labels,1),dtype=tf.float32)
        tf_out_neg_mask = tf.cast(tf.equal(tf_labels,-1),dtype=tf.float32)
        tf_out_mask = tf_out_pos_mask + tf_out_neg_mask

        loss = tf.reduce_mean(tf.reduce_sum(((tf_out-tf_labels)**2)*tf_out_mask,axis=[1,2,3]))


    elif method == 'weighted':
        tf_out = get_inference(tf_inputs, OUTPUT_SHAPES)

        tf_pos_mask = tf.cast(tf.equal(tf_labels, 1), dtype=tf.float32)

        tf_out_neg_rand_mask = tf.cast(
            tf.greater(tf.truncated_normal(tf_labels.get_shape().as_list(), dtype=tf.float32), 1.0), dtype=tf.float32)
        tf_neg_mask = tf.cast(tf.equal(tf_labels, -1), dtype=tf.float32) * tf_out_neg_rand_mask

        tf_pos_mask_weighted = tf.cast(tf.equal(tf_labels, 1), dtype=tf.float32) * 0.95
        tf_neg_mask_weighted = tf.cast(tf.equal(tf_labels, -1), dtype=tf.float32) * tf_out_neg_rand_mask * 0.95 # TODO: 0.1 bias?

        tf_outer_pos_mask = tf.cast(tf.equal(tf_labels, 1), dtype=tf.float32)*5.0 + \
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
                    loss = loss + beta * tf.reduce_sum(tf.get_variable(TF_WEIGHTS_SCOPE)**2)

    elif method == 'equal_number_of_posneg_samples':
        tf_out = get_inference(tf_inputs, OUTPUT_SHAPES)

        tf_out_pos_mask = tf.cast(tf.equal(tf_labels, 1), dtype=tf.float32)
        tf_out_neg_mask = tf.cast(tf.equal(tf_labels, -1), dtype=tf.float32)
        tf_out_neg_rand_mask = tf.cast(tf.greater(tf.truncated_normal(tf_labels.get_shape().as_list(),dtype=tf.float32),0.0),dtype=tf.float32)
        tf_out_neg_mask = tf_out_neg_mask*tf_out_neg_rand_mask
        tf_out_mask = tf_out_pos_mask + tf_out_neg_mask
        loss = tf.reduce_mean(tf.reduce_sum(((tf_out - tf_labels) ** 2) , axis=[1, 2, 3]))

    return loss


def optimize_model(loss,global_step):
    learning_rate = tf.cond(global_step<20,lambda: tf.minimum(tf.train.exponential_decay(0.00001,global_step,2,1.01,staircase=True),1e-4),
                            lambda: tf.maximum(tf.train.exponential_decay(0.00001,global_step,2,0.99,staircase=True),1e-7))
    optimize = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss,global_step)
    #optimize = tf.train.GradientDescentOptimizer(learning_rate=0.000001).minimize(loss)
    return optimize,learning_rate


test_width = 600
test_height = 300
pred_data_dir = 'pred_data'

normalize_constant = 150

if __name__ == '__main__':
    global sess,graph

    file_count = 1300
    data_folder = 'data/'

    width = 60
    height = 30

    channels = 2

    with sess.as_default() and graph.as_default():
        tf_inpts = tf.placeholder(dtype=tf.float32, shape=[batch_size, height, width, channels], name='inputs')
        tf_labls = tf.placeholder(dtype=tf.float32, shape=[batch_size, height, width, 1], name='labels')
        global_step = tf.Variable(0,dtype=tf.int32,trainable=False)
        tf_test_inputs = tf.placeholder(dtype=tf.float32, shape=[1, height, width, channels], name='inputs')
        build_tensorflw_variables()

        if OUTPUT_TYPE=='classification':
            tf_loss = calculate_loss_one_hot(tf_inpts,tf_labls)
            tf_prediction = get_predictions_with_ohe(tf_test_inputs)

        elif OUTPUT_TYPE=='regression':
            tf_loss = calculate_loss(tf_inpts, tf_labls)
            tf_prediction = get_prediction(tf_test_inputs)
        else:
            raise NotImplementedError

        tf_optimize,tf_learning_rate = optimize_model(tf_loss,global_step)

        tf.global_variables_initializer().run()
        for epoch in range(1000):
            avg_loss = []
            for step in range(file_count//batch_size):
                inp1, lbl1 = load_data.load_batch_npz(data_folder, batch_size, height, width, channels, file_count,shuffle=False)
                occupied_size  = np.where(lbl1==1)[0].size
                #print('occupied ratio: ',occupied_size/lbl1.size)
                l, labels, _ = sess.run([tf_loss, tf_labls, tf_optimize], feed_dict={tf_inpts:inp1/normalize_constant, tf_labls:lbl1})
                avg_loss.append(l)
                #print(l)

            if (epoch+1)%1==0:

                xx, yy = np.meshgrid(np.arange(-299, 300, 2), np.arange(1, 300, 2))
                test_full_map = -100 + np.zeros(xx.shape)

                for col_no in range(0, 150, 30):
                    for row_no in range(0, 300, 60):

                        local_xx = xx[col_no:col_no + 30, row_no:row_no + 60][:, :, np.newaxis]
                        local_yy = yy[col_no:col_no + 30, row_no:row_no + 60][:, :, np.newaxis]
                        test_input = np.concatenate((local_xx, local_yy), axis=2)[np.newaxis, :, :, :]

                        pred = sess.run(tf_prediction, feed_dict={tf_test_inputs:test_input/normalize_constant})
                        test_full_map[col_no:col_no + 30, row_no:row_no + 60] = pred[0, :, :,0]

                plt.close('all')
                plt.figure(figsize=(12, 10))
                plt.subplot(211)
                plt.title('Occupancy $\in [elastic, elastic]$')
                plt.scatter(xx.ravel(), yy.ravel(), c=test_full_map.ravel(), s=1, cmap='jet')
                plt.colorbar()
                plt.xlim([-300, 300]); plt.ylim([0, 300])
                plt.axis('equal')
                plt.subplot(212)
                plt.title('Occupancy $\in [-1, 1]$')
                plt.scatter(xx.ravel(), yy.ravel(), c=test_full_map.ravel(), s=1, cmap='jet', vmin=-1, vmax=1)
                plt.colorbar()
                plt.xlim([-300, 300]); plt.ylim([0, 300])
                plt.axis('equal')
                #plt.show()
                plt.savefig(pred_data_dir+os.sep+'pred_%d.png'%(epoch+1))

            print('Loss Epoch (%d): %.5f'%(epoch,np.mean(avg_loss)))
            print('Learning rate: ',sess.run(tf_learning_rate))
