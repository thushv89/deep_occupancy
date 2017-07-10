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
batch_size = 25
VAR_SHAPES = {'conv1': [3,3,2,32],'conv2':[3,3,32,64],'deconv2':[3,3,32,64],'deconv1':[3,3,1,32]}
OUTPUT_SHAPES = {'deconv2':[batch_size,30,60,32],'deconv1':[batch_size,30,60,1]}
TEST_OUTPUT_SHAPES = {'deconv2':[1,30,60,32],'deconv1':[1,30,60,1]}
POOL_STRIDES = {'pool1':[1,2,2,1],'pool2':[1,3,3,1]}
TF_WEIGHTS_SCOPE = 'weights'
TF_BIAS_SCOPE = 'bias'
TF_DECONV_SCOPE = 'deconv'

graph = tf.get_default_graph()
sess = tf.InteractiveSession(graph=graph)


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def build_tensorflw_variables():
    '''
    Build the required tensorflow variables to populate the VGG-16 model
    All are initialized with zeros (initialization doesn't matter in this case as we assign exact values later)
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
                    h = lrelu(
                        tf.nn.conv2d(tf_inputs, weight, strides=[1,1,1,1], padding='SAME') + bias,
                        name='hidden')
                    print('\t\t\tOutput shape: ', h.get_shape().as_list())
                else:
                    h = lrelu(tf.nn.conv2d(h, weight, strides=[1,1,1,1], padding='SAME') + bias,
                                   name='hidden')
            if scope.startswith('deconv'):
                weight, bias = tf.get_variable(TF_WEIGHTS_SCOPE), tf.get_variable(TF_BIAS_SCOPE)

                if si== len(CONV_SCOPES)-1:
                    print('\t\tConvolution with TanH activation for ', scope)
                    h = tf.nn.tanh(tf.nn.conv2d_transpose(h, weight,OUTPUT_SHAPES[scope],strides=[1,1,1,1],padding="SAME") + bias)
                    print('\t\t\tOutput shape: ', h.get_shape().as_list())

                else:
                    print('\t\tConvolution with ReLU activation for ', scope)
                    h = lrelu(tf.nn.conv2d_transpose(h, weight,OUTPUT_SHAPES[scope],strides=[1,1,1,1],padding="SAME")+bias)
                    print('\t\t\tOutput shape: ', h.get_shape().as_list())

            if 'pool' in scope:
                print('\t\tMax Pooling for %s', scope)
                h = tf.nn.max_pool(h,[1,3,3,1],POOL_STRIDES[scope],padding='SAME')
                print('\t\t\tOutput shape: ',h.get_shape().as_list())

    return h

def get_prediction(tf_input):
    tf_out = get_inference(tf_input,TEST_OUTPUT_SHAPES)
    return tf_out


def calculate_loss(tf_inputs,tf_labels):
    tf_out = get_inference(tf_inputs,OUTPUT_SHAPES)

    tf_out_pos_mask = tf.cast(tf.equal(tf_labels,1),dtype=tf.float32)
    tf_out_neg_mask = tf.cast(tf.equal(tf_labels,-1),dtype=tf.float32)*0.1
    tf_out_mask = tf_out_pos_mask + tf_out_neg_mask
    #tf_out_mask = tf.cast(tf.not_equal(tf_labels,0),dtype=tf.float32)
    loss = tf.reduce_mean(tf.reduce_sum(((tf_out-(tf_labels*tf_out_neg_mask))**2)*tf_out_mask,axis=[1,2,3]))
    #loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf_out,labels=tf_labels),axis=[1,2,3]))
    return loss,tf_out_mask,tf_labels


def optimize_model(loss):

    optimize = tf.train.MomentumOptimizer(learning_rate=0.0001,momentum=0.9).minimize(loss)
    return optimize


test_width = 600
test_height = 300
pred_data_dir = 'pred_data'

if __name__ == '__main__':
    global sess,graph

    file_count = 1300
    data_folder = 'data'

    width = 60
    height = 30

    channels = 2

    with sess.as_default() and graph.as_default():
        tf_inpts = tf.placeholder(dtype=tf.float32, shape=[batch_size, height, width, channels], name='inputs')
        tf_labls = tf.placeholder(dtype=tf.float32, shape=[batch_size, height, width,1], name='labels')
        tf_test_inputs = tf.placeholder(dtype=tf.float32, shape=[1, height, width, channels], name='inputs')
        build_tensorflw_variables()

        tf_loss,tf_mask,_ = calculate_loss(tf_inpts, tf_labls)
        tf_optimize = optimize_model(tf_loss)
        tf_prediction = get_prediction(tf_test_inputs)
        tf.global_variables_initializer().run()
        for epoch in range(50):
            avg_loss = []
            min,max = 1000,-1000
            for step in range(20):
                inp1,lbl1 = load_data.load_batch_npz(data_folder,batch_size,height,width,channels,file_count)
                #norm_inp1 = inp1[:,:,:,0]/
                l,mask,labels,_= sess.run([tf_loss,tf_mask,tf_labls,tf_optimize],feed_dict={tf_inpts:inp1/300,tf_labls:lbl1})
                avg_loss.append(l)

                '''plt.figure(1)
                plt.subplot(121)
                plt.imshow(mask[0,:,:,0])

                print(mask[0,:,:,0])
                print('blah')
                print(labels[0,:,:,0])

                plt.subplot(122)
                plt.imshow(labels[0,:,:,0])
                plt.colorbar()
                plt.savefig('test_labels.png')

                sys.exit(1)'''

                if np.min(inp1) < min:
                    min = np.min(inp1)
                if np.max(inp1) > max:
                    max = np.max(inp1)
            print(min)
            print(max)
            if (epoch+1)%1==0:

                test_full_map = np.empty((test_height,test_width),dtype=np.float32)

                for r_i in range(test_height//height):
                    for c_i in range(-test_width//(width*2),test_width//(width*2)):
                        test_input = np.empty((1,30, 60, 2), dtype=np.float32)
                        for rr_i in range(30):
                            test_input[0,rr_i,:,0] = (np.ones((60),dtype=np.float32)*((r_i*30)+rr_i))/300.0
                        for cc_i in range(60):
                            test_input[0,:, cc_i, 0] = (np.ones((30),dtype=np.float32)*((c_i*60)+cc_i))/300.0

                        pred = sess.run(tf_prediction,feed_dict={tf_test_inputs:test_input})
                        #print(test_input)
                        #if r_i==0 and c_i==0:
                            #print(pred)
                        #sys.exit(1)
                        #plt.close('all')
                        #plt.imshow(pred[0, :, :, 0],cmap='jet',vmin=-1,vmax=1)
                        #plt.colorbar()
                        #plt.savefig(pred_data_dir+os.sep+'pred_sub_%d_%d_%d.png' % (epoch,r_i,c_i))
                        im = img_as_uint(pred[0,:,:,0])
                        io.imsave(pred_data_dir+os.sep+'pred_sub_%d_%d_%d.png' % (epoch,r_i,c_i),im)
                        print(r_i,(c_i+(test_width//(width*2)))*60,(c_i+(test_width//(width*2))+1)*60)
                        test_full_map[r_i*30:(r_i+1)*30,(c_i+(test_width//(width*2)))*60:(c_i+(test_width//(width*2))+1)*60] = pred[0,:,:,0]

                plt.close('all')
                fig = plt.figure(epoch)
                plt.imshow(test_full_map, cmap='jet', vmin=-1, vmax=1)
                plt.colorbar()
                plt.savefig(pred_data_dir+os.sep+'pred_%d.png'%(epoch+1))

            print('Loss Epoch (%d): %.5f'%(epoch,np.mean(avg_loss)))
