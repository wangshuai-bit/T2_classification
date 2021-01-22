# try to print the error image 20190520
import tensorflow as tf
import pickle
import time
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope
from PIL import Image
from load_data import *
import matplotlib.pyplot as plt
from tensorflow.python import pywrap_tensorflow
import math

from itertools import cycle
from sklearn.metrics import roc_curve,auc
from scipy import interp

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# Hyperparameter
growth_k = 24
nb_block = 2  # how many (dense block + Transition Layer) ?
#init_learning_rate = 2.0e-4
init_learning_rate = 0.01
#tmp_learning_rate = 0.1
init_lamda_1 = 0.00
init_lamda_2 = 1.00
epsilon = 1e-4  # AdamOptimizer epsilon
dropout_rate = 0.30
keep_prob = 1.0

# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 8e-4
weight_decay_l1 = 0

# Label & batch_size
batch_size = 32
dataset_size = 4800
iteration = 150
# batch_size * iteration = data_set_number

test_iteration = 10

# total_epochs = 300
total_epochs = 300

# regularzer
reg_scale = 0.4

# train
isTrain =False

#datasets
datasets = {}


root_path = '/home/wangshuai/ckpts_for_zhengyao/pt_5_to_2_lr_transfer_test_5_5to2'
#os.mkdir(root_path)
txt_path = os.path.join(root_path, 'logs.txt')
print(txt_path)
ckpt_path = root_path
summary_path = root_path
save_path = os.path.join(root_path, 'train_64_pt_5_to_2')
write_title = "train_64_pt_5_to_2, init_learning_rate:%.6f, dropout_rate:%.2f, " \
              "weight_decay%.4f,total_epochs%.4f, batch_size%.1f\n" \
              % (init_learning_rate,dropout_rate,weight_decay,total_epochs,batch_size)



start_time = time.time()
print("start time is", start_time)


def variable_summaries(var,name):
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        tf.summary.histogram('histogram', var)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def Global_Average_Pooling(x, stride=1):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    It is global average pooling without tflearn
    """

    return global_avg_pool(x, name='Global_avg_pooling')
    # But maybe you need to install h5py and curses or not


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=False):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def Drop_out(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def Relu(x):
    return tf.nn.relu(x)


def Average_pooling(x, pool_size=[2, 2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def max_pool(input, k_size=1, stride=1, name=None):
    return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1],
                          padding='SAME', name=name)

def Concatenation(layers):
    return tf.concat(layers, axis=3)


def Linear(x):
    #dropout = tf.layers.dropout(inputs=x, rate=0.2, training=training_flag)  # add dropout here

    #relu_1 = tf.nn.relu(x)

    dense_1 = tf.layers.dense(inputs=x, units=10, name='linear_1', use_bias=True,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale))


    dense = tf.layers.dense(inputs=dense_1, units=class_num, name='linear_5', use_bias=True,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale))


    return dense



def Evaluate(sess, epoch):
    test_acc = 0.0
    test_acc_norm = 0.0
    test_acc_arc = 0.0
    test_loss = 0.0
    test_pre_index = 0
    train_pre_index = 0
    add = 67
    #add = 930
    y_amount_0 = 0
    y_amount_1 = 0
    y_amount_2 = 0
    y_amount_3 = 0

    equal = 0
    y_equal_0 = 0
    y_equal_1 = 0
    y_equal_2 = 0
    y_equal_3 = 0

    y_0to1 = 0
    y_0to2 = 0
    y_0to3 = 0
    y_1to0 = 0
    y_1to2 = 0
    y_1to3 = 0
    y_2to0 = 0
    y_2to1 = 0
    y_2to3 = 0
    y_3to0 = 0
    y_3to1 = 0
    y_3to2 = 0


    y_equal_0_pro_sigmoid = 0
    y_equal_1_pro_sigmoid = 0

    y_equal_0_pro_softmax = 0
    y_equal_1_pro_softmax = 0
    y_equal_2_pro_softmax = 0
    y_equal_3_pro_softmax = 0


    y_all_1_pro_sigmoid = 0
    y_all_0_pro_sigmoid = 0


    y_all_0_pro_softmax = 0
    y_all_1_pro_softmax = 0
    y_all_2_pro_softmax = 0
    y_all_3_pro_softmax = 0

    y_equal_0_pro_sigmoid_wrong = 0
    mid = 0
    mid_1 = 0
    y_score = np.empty(shape=[0, 4])
    y_onehot = np.empty(shape=[0, 4])

    for it in range(test_iteration):

        test_batch_x = test_x[test_pre_index: test_pre_index + add]
        test_batch_y = test_y[test_pre_index: test_pre_index + add]
        test_batch_p = test_p[test_pre_index: test_pre_index + add]
        test_pre_index = test_pre_index + add


        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            path: test_batch_p,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)
        '''
        logits_watch = sess.run(logits, feed_dict=test_feed_dict)
        print("logit is", logits_watch)
        print("label is ", test_batch_y)
        '''
        if epoch >= total_epochs-1:
            result_one = sess.run(logits, feed_dict=test_feed_dict)
            loss_, acc_= sess.run([cost, accuracy], feed_dict=test_feed_dict)

            y_score = np.append(y_score, result_one, axis=0)
            y_onehot = np.append(y_onehot, test_batch_y, axis=0)

        test_loss += loss_ / 10.0
        test_acc += acc_ / 10.0

        if epoch >= total_epochs-1:
            # print("the acc of this time is ", acc_)
            # print("the all acc is ", test_acc)

            result_one_sigmoid = sess.run(tf.nn.sigmoid(result_one))
            result_one_softmax = sess.run(tf.nn.softmax(result_one))
            result_one_argmax = sess.run(tf.argmax(result_one, 1))
            test_batch_y_argmax = sess.run(tf.argmax(test_batch_y, 1))
            path_one = test_batch_p

            for i in range(len(test_batch_y_argmax)):
                if test_batch_y_argmax[i] == 0:
                    y_amount_0 = y_amount_0 + 1
                    y_all_0_pro_softmax = y_all_0_pro_softmax + result_one_softmax[i]
                    if result_one_argmax[i] == 1:
                        y_0to1 = y_0to1 + 1
                        #print("y_0to1 is ", path_one[i])
                    elif result_one_argmax[i] == 2:
                        y_0to2 = y_0to2 + 1
                        #print("y_0to2 is ", path_one[i])
                    elif result_one_argmax[i] == 3:
                        y_0to3 = y_0to3 + 1
                        #print("y_0to3 is ", path_one[i])
                    elif result_one_argmax[i] == test_batch_y_argmax[i]:
                        y_equal_0 = y_equal_0 + 1
                        y_equal_0_pro_sigmoid = y_equal_0_pro_sigmoid + result_one_sigmoid[i]
                        y_equal_0_pro_softmax = y_equal_0_pro_softmax + result_one_softmax[i]
                        #print("0 is", path_one[i])
                elif test_batch_y_argmax[i] == 1:
                    y_amount_1 = y_amount_1 + 1
                    y_all_1_pro_softmax = y_all_1_pro_softmax + result_one_softmax[i]
                    if result_one_argmax[i] == 0:
                        y_1to0 = y_1to0 + 1
                        #print("y_1to0 is", path_one[i])
                    elif result_one_argmax[i] == 2:
                        y_1to2 = y_1to2 + 1
                        #print("y_1to2 is", path_one[i])
                    elif result_one_argmax[i] == 3:
                        y_1to3 = y_1to3 + 1
                        #print("y_1to3 is", path_one[i])
                    elif result_one_argmax[i] == test_batch_y_argmax[i]:
                        y_equal_1 = y_equal_1 + 1
                        y_equal_1_pro_sigmoid = y_equal_1_pro_sigmoid + result_one_sigmoid[i]
                        y_equal_1_pro_softmax = y_equal_1_pro_softmax + result_one_softmax[i]
                        #print("1 is", path_one[i])
                elif test_batch_y_argmax[i] == 2:
                    y_amount_2 = y_amount_2 + 1
                    y_all_2_pro_softmax = y_all_2_pro_softmax + result_one_softmax[i]
                    if result_one_argmax[i] == 0:
                        y_2to0 = y_2to0 + 1
                        #print("y_2to0 is", path_one[i])
                    elif result_one_argmax[i] == 1:
                        y_2to1 = y_2to1 + 1
                        #print("y_2to1 is", path_one[i])
                    elif result_one_argmax[i] == 3:
                        y_2to3 = y_2to3 + 1
                        #print("y_2to3 is", path_one[i])
                    if result_one_argmax[i] == test_batch_y_argmax[i]:
                        y_equal_2 = y_equal_2 + 1
                        y_equal_2_pro_softmax = y_equal_2_pro_softmax + result_one_softmax[i]
                        #print("2 is" , path_one[i])
                elif test_batch_y_argmax[i] == 3:
                    y_amount_3 = y_amount_3 + 1
                    y_all_3_pro_softmax = y_all_3_pro_softmax + result_one_softmax[i]
                    if result_one_argmax[i] == 0:
                        y_3to0 = y_3to0 + 1
                        #print("y_3to0 is", path_one[i])
                    elif result_one_argmax[i] == 1:
                        y_3to1 = y_3to1 + 1
                        #print("y_3to1 is", path_one[i])
                    elif result_one_argmax[i] == 2:
                        y_3to2 = y_3to2 + 1
                        #print("y_3to2 is", path_one[i])

                    elif result_one_argmax[i] == test_batch_y_argmax[i]:
                        y_equal_3 = y_equal_3 + 1
                        y_equal_3_pro_softmax = y_equal_3_pro_softmax + result_one_softmax[i]
                        #print("3 is", path_one[i])

            # print("the result_one_argmax is ", result_one_argmax)
            # print("the test_batch_y_argmax is ", test_batch_y_argmax)
            # print("result_one_softmax is ", result_one_softmax)
            # print("test_batch_y is ", test_batch_y)
    if epoch >=total_epochs-1:
        print("y_score and y_onehot shape is ", y_score.shape, y_onehot.shape)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(class_num):
            fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # first aggregate all the false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(class_num)]))
        # then interpolate all ROC curves at this point
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(class_num):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # finally average it and compute AUC
        mean_tpr /= class_num
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        fpr_macro = fpr["macro"]
        tpr_macro = tpr["macro"]
        roc_auc_macro = roc_auc["macro"]
        # plot all ROC curves
        subtype = ["ccRCC","CRCC","AML","PRCC"]
        plt.plot(fpr["macro"], tpr["macro"], label="macro-average ROC curve(area = {0:0.2f})".format(roc_auc["macro"]),
                 color="navy", linestyle=":", linewidth=4)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(class_num), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label="ROC curve of {0}(area = {1:0.2f})".format(subtype[i], roc_auc[i]))
        print(fpr[0].shape)
        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.title("ROC to multi-classification")
        plt.legend(loc="lower right")
        plt.savefig("ROC of 5_to_2.jpg")
        plt.show()



    if epoch >= total_epochs-1:
        print("the amount of 0 is and the equal is ", y_amount_0, y_equal_0)
        print("the amount of 1 is and the equal is ", y_amount_1, y_equal_1)
        print("the amount of 2 is and the equal is ", y_amount_2, y_equal_2)
        print("the amount of 3 is and the equal is ", y_amount_3, y_equal_3)
        print("the equal pro of 0 is sigmoid, softmax", y_equal_0_pro_sigmoid / y_amount_0,
              y_equal_0_pro_softmax / y_amount_0)
        print("the equal pro of 1 is sigmoid, softmax", y_equal_1_pro_sigmoid / y_amount_1,
              y_equal_1_pro_softmax / y_amount_1)
        print("the equal pro of 2 is sigmoid, softmax", y_equal_2_pro_softmax / y_amount_2)
        print("the equal pro of 3 is sigmoid, softmax", y_equal_3_pro_softmax / y_amount_3)
        print("the all pro of 0 is sigmoid ", y_all_0_pro_softmax / y_amount_0)
        print("the all pro of 1 is sigmoid ", y_all_1_pro_softmax / y_amount_1)
        print("the all pro of 0 is sigmoid ", y_all_2_pro_softmax / y_amount_2)
        print("the all pro of 1 is sigmoid ", y_all_3_pro_softmax / y_amount_3)

        # print("the pro of 0 wrong is , and the mid0, mid1 is ", y_equal_0_pro_sigmoid_wrong/(y_amount_0-y_equal_0), mid, mid_1)
        y_0_acc = y_equal_0 / y_amount_0
        y_1_acc = y_equal_1 / y_amount_1
        y_2_acc = y_equal_2 / y_amount_2
        y_3_acc = y_equal_3 / y_amount_3
        print("the acc of 0 is ", y_0_acc)
        print("the acc of 1 is ", y_1_acc)
        print("the acc of 2 is ", y_2_acc)
        print("the acc of 3 is ", y_3_acc)

        print("the 0 class is", y_equal_0, y_0to1, y_0to2, y_0to3, y_amount_0)
        print("the 1 class is", y_1to0, y_equal_1, y_1to2, y_1to3, y_amount_1)
        print("the 2 class is", y_2to0, y_2to1, y_equal_2, y_2to3, y_amount_2)
        print("the 3 class is", y_3to0, y_3to1, y_3to2, y_equal_3, y_amount_3)

        print("the precision of 0,1,2,3", "0", (y_equal_0 + y_1to0 + y_2to0 + y_3to0),
              y_equal_0 / (y_equal_0 + y_1to0 + y_2to0 + y_3to0),
              "1", (y_0to1 + y_equal_1 + y_2to1 + y_3to1), y_equal_1 / (y_0to1 + y_equal_1 + y_2to1 + y_3to1),
              "2", (y_0to2 + y_1to2 + y_equal_2 + y_3to2), y_equal_2 / (y_0to2 + y_1to2 + y_equal_2 + y_3to2),
              "3", (y_0to3 + y_1to3 + y_2to3 + y_equal_3), y_equal_3 / (y_0to3 + y_1to3 + y_2to3 + y_equal_3)
              )


    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary


class RESNet():
    def __init__(self, x, training, labels):
        self.training = training
        self.model = self.ResNet18(x, is_training=training, pooling_and_fc=True,
                 reuse=False, kernel_initializer = tf.contrib.layers.variance_scaling_initializer())


    def identity_block2d(self,input_tensor, kernel_size, filters, stage, block, is_training, reuse,
                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):
        filters1, filters2, filters3 = filters

        conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
        bn_name_2 = 'bn' + str(stage) + '_' + str(block) + '_3x3'
        x = tf.layers.conv2d(input_tensor, filters2, kernel_size, use_bias=False, padding='SAME',
                             kernel_initializer=kernel_initializer, name=conv_name_2, reuse=reuse)
        x = Batch_Normalization(x, training=is_training, scope=bn_name_2)
        x = tf.nn.relu(x)

        conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
        bn_name_3 = 'bn' + str(stage) + '_' + str(block) + '_1x1_increase'
        x = tf.layers.conv2d(x, filters3, (kernel_size, kernel_size), use_bias=False, padding='SAME',
                             kernel_initializer=kernel_initializer, name=conv_name_3, reuse=reuse)
        x = Batch_Normalization(x, training=is_training, scope=bn_name_3)

        x = tf.add(input_tensor, x)
        x = tf.nn.relu(x)
        return x

    def conv_block_2d(self,input_tensor, kernel_size, filters, stage, block, is_training, reuse, strides=(2, 2),
                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):
        filters1, filters2, filters3 = filters

        conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
        bn_name_2 = 'bn' + str(stage) + '_' + str(block) + '_3x3'
        x = tf.layers.conv2d(input_tensor, filters2, (kernel_size, kernel_size), use_bias=False, strides=strides,
                             padding='SAME', kernel_initializer=kernel_initializer, name=conv_name_2, reuse=reuse)
        x = Batch_Normalization(x, training=is_training, scope=bn_name_2)
        x = tf.nn.relu(x)

        conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
        bn_name_3 = 'bn' + str(stage) + '_' + str(block) + '_1x1_increase'
        x = tf.layers.conv2d(x, filters3, (kernel_size, kernel_size), use_bias=False, padding='SAME',
                             kernel_initializer=kernel_initializer, name=conv_name_3, reuse=reuse)
        x = Batch_Normalization(x, training=is_training, scope=bn_name_3)

        conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_shortcut'
        bn_name_4 = 'bn' + str(stage) + '_' + str(block) + '_1x1_shortcut'
        shortcut = tf.layers.conv2d(input_tensor, filters3, (kernel_size, kernel_size), use_bias=False, strides=strides,
                                    padding='SAME', kernel_initializer=kernel_initializer, name=conv_name_4,
                                    reuse=reuse)
        shortcut = Batch_Normalization(shortcut, training=is_training, scope=bn_name_4)

        x = tf.add(shortcut, x)
        x = tf.nn.relu(x)
        return x

    def ResNet18(self,input_tensor, is_training=True, pooling_and_fc=True, reuse=False,
                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):
        print("the input_tensor is ", input_tensor)
        input_tensor_tile = tf.tile(input_tensor, [1,1,1,3])
        print("after tf.tile, the input tensor is", input_tensor_tile)
        x = tf.layers.conv2d(input_tensor_tile, 32, (3, 3), strides=(1, 1), kernel_initializer=kernel_initializer,
                             use_bias=False, padding='SAME', name='conv1_1/3x3_s1', reuse=reuse)
        x = Batch_Normalization(x, training=is_training,scope ='bn1_1/3x3_s1')
        x = tf.nn.relu(x)

        x1 = self.identity_block2d(x, 3, [48, 32, 32], stage=2, block='1b', is_training=is_training, reuse=reuse,
                              kernel_initializer=kernel_initializer)
        x1 = self.identity_block2d(x1, 3, [48, 32, 32], stage=3, block='1c', is_training=is_training, reuse=reuse,
                              kernel_initializer=kernel_initializer)

        x2 = self.conv_block_2d(x1, 3, [96, 64, 64], stage=3, block='2a', strides=(2, 2), is_training=is_training,
                           reuse=reuse, kernel_initializer=kernel_initializer)
        x2 = self.identity_block2d(x2, 3, [96, 64, 64], stage=3, block='2b', is_training=is_training, reuse=reuse,
                              kernel_initializer=kernel_initializer)

        x3 = self.conv_block_2d(x2, 3, [128, 128, 128], stage=4, block='3a', strides=(2, 2), is_training=is_training,
                           reuse=reuse, kernel_initializer=kernel_initializer)
        x3 = self.identity_block2d(x3, 3, [128, 128, 128], stage=4, block='3b', is_training=is_training, reuse=reuse,
                              kernel_initializer=kernel_initializer)

        x4 = self.conv_block_2d(x3, 3, [256, 256, 256], stage=5, block='4a', strides=(2, 2), is_training=is_training,
                           reuse=reuse, kernel_initializer=kernel_initializer)
        x4 = self.identity_block2d(x4, 3, [256, 256, 256], stage=5, block='4b', is_training=is_training, reuse=reuse,
                              kernel_initializer=kernel_initializer)

        # print('before gap: ', x4)
        x4 = tf.reduce_mean(x4, [1, 2])
        x4 = Drop_out(x4, dropout_rate, is_training)
        # print('after gap: ', x4)
        # flatten = tf.contrib.layers.flatten(x4)
        prob = tf.layers.dense(x4, 4, reuse=reuse, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               use_bias=True, name="fully_connected")


        return prob

    def ResNet34(self, input_tensor, is_training, pooling_and_fc=True,
                 reuse=False, kernel_initializer = tf.contrib.layers.variance_scaling_initializer()):
        x = tf.layers.conv2d(input_tensor, 32, (5, 5), strides=(1, 1), kernel_initializer=kernel_initializer,
                             use_bias=False, padding='SAME', name='conv1_1/3x3_s1', reuse=reuse)
        x = Batch_Normalization(x, training=is_training,scope ='bn1_1/3x3_s1')
        x = tf.nn.relu(x)
        variable_summaries(x, name='x_0')

        x1 = self.identity_block2d(x, 3, [48, 32, 32], stage=1, block='1a', is_training=is_training, reuse=reuse,
                              kernel_initializer=kernel_initializer)
        x1 = self.identity_block2d(x1, 3, [48, 32, 32], stage=1, block='1b', is_training=is_training, reuse=reuse,
                              kernel_initializer=kernel_initializer)
        x1 = self.identity_block2d(x1, 3, [48, 32, 32], stage=1, block='1c', is_training=is_training, reuse=reuse,
                              kernel_initializer=kernel_initializer)
        variable_summaries(x1, name='x_1')

        x2 = self.conv_block_2d(x1, 3, [96, 64, 64], stage=2, block='2a', strides=(2, 2), is_training=is_training,
                           reuse=reuse, kernel_initializer=kernel_initializer)
        x2 = self.identity_block2d(x2, 3, [96, 64, 64], stage=2, block='2b', is_training=is_training, reuse=reuse,
                              kernel_initializer=kernel_initializer)
        x2 = self.identity_block2d(x2, 3, [96, 64, 64], stage=2, block='2c', is_training=is_training, reuse=reuse,
                              kernel_initializer=kernel_initializer)
        x2 = self.identity_block2d(x2, 3, [96, 64, 64], stage=2, block='2d', is_training=is_training, reuse=reuse,
                              kernel_initializer=kernel_initializer)
        variable_summaries(x2, name='x_2')

        x3 = self.conv_block_2d(x2, 3, [128, 128, 128], stage=3, block='3a', strides=(2, 2), is_training=is_training,
                           reuse=reuse, kernel_initializer=kernel_initializer)
        x3 = self.identity_block2d(x3, 3, [128, 128, 128], stage=3, block='3b', is_training=is_training, reuse=reuse,
                              kernel_initializer=kernel_initializer)
        x3 = self.identity_block2d(x3, 3, [128, 128, 128], stage=3, block='3c', is_training=is_training, reuse=reuse,
                              kernel_initializer=kernel_initializer)
        x3 = self.identity_block2d(x3, 3, [128, 128, 128], stage=3, block='3d', is_training=is_training, reuse=reuse,
                              kernel_initializer=kernel_initializer)
        x3 = self.identity_block2d(x3, 3, [128, 128, 128], stage=3, block='3e', is_training=is_training, reuse=reuse,
                              kernel_initializer=kernel_initializer)
        x3 = self.identity_block2d(x3, 3, [128, 128, 128], stage=3, block='3f', is_training=is_training, reuse=reuse,
                              kernel_initializer=kernel_initializer)
        variable_summaries(x3, name='x_3')

        x4 = self.conv_block_2d(x3, 3, [256, 256, 256], stage=4, block='4a', strides=(2, 2), is_training=is_training,
                           reuse=reuse, kernel_initializer=kernel_initializer)
        x4 = self.identity_block2d(x4, 3, [256, 256, 256], stage=4, block='4b', is_training=is_training, reuse=reuse,
                              kernel_initializer=kernel_initializer)
        x4 = self.identity_block2d(x4, 3, [256, 256, 256], stage=4, block='4c', is_training=is_training, reuse=reuse,
                              kernel_initializer=kernel_initializer)

        # print('before gap: ', x4)
        x4 = tf.reduce_mean(x4, [1, 2])

        x4 = Drop_out(x4, dropout_rate, is_training)

        # print('after gap: ', x4)
        # flatten = tf.contrib.layers.flatten(x4)
        prob = tf.layers.dense(x4, 4, reuse=reuse, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1),
                               name="fully_connected")

        return prob



#train_x_pre, train_y_pre, test_x_pre, test_y_pre = prepare_data(train_files = '/training_64_4class_pk.pickle', test_files = '/test_64_4class_pk.pickle')
train_x, train_y, train_p, test_x, test_y, test_p = prepare_data(train_files = '/train_64_pt_all_sel_5_to_2', test_files = '/test_64_pt_all_sel_5_to_2')

train_x, test_x = color_preprocessing(train_x, test_x)
print("after select,the shape of train data and label is  ", train_x.shape, train_y.shape)
print("aftre select, the shape of test data and label is ", test_x.shape, test_y.shape)




# image_size = 32, img_channels = 3, class_num = 10 in cifar10
x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
label = tf.placeholder(tf.float32, shape=[None, class_num])
path = tf.placeholder(tf.string)
training_flag = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')


logits = RESNet(x=x, training=training_flag, labels=label).model
#logits, cos_t, s_train, logits_2, logits_3, logits_4, logits_5, logits_6, center_loss= DenseNet(x=x, nb_blocks=nb_block, filters=growth_k, training=training_flag, labels = label).model

# reg_ws = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"DenseNet")

# weights_regularizer = tf.contrib.layers.l1_regularizer(0.4)
reg_ws = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'DenseNet')
print("label", label, "logits", logits)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
l1 = tf.add_n([tf.contrib.layers.l1_regularizer(0.5)(var) for var in tf.trainable_variables()])


"""
l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=nesterov_momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss * weight_decay)

In paper, use MomentumOptimizer
init_learning_rate = 0.1

but, I'll use AdamOptimizer
"""
cost = cross_entropy + L_metric_l2_regularizer
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)


tr_vars = tf.trainable_variables()
var_list = [t for t in tr_vars]
print("type of var is ", type(var_list),var_list)
i=0
for ttt in var_list:
    print("t is ", i,ttt)
    i+=1
new_var_list = var_list[30:]


optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
train = optimizer.minimize(cost + l2 * weight_decay )

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#merge all the summary

restore_variable_list = tf.contrib.framework.get_variables_to_restore(exclude=["fully_connected","is_training"])
saver = tf.train.Saver(restore_variable_list)
saver_2 = tf.train.Saver()
with open(txt_path, 'a') as f:
    f.write(write_title)
    print("write finished,\n")

with tf.Session() as sess:
    '''  '''
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        reader = pywrap_tensorflow.NewCheckpointReader(save_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            print("tensor name",key)
        sess.run(tf.global_variables_initializer())
        saver_2.restore(sess, ckpt.model_checkpoint_path)
        print("load the model")

    else :
        sess.run(tf.global_variables_initializer())
        print("new initial")

    #sess.run(tf.global_variables_initializer())
    #print("new initial")

    summary_writer = tf.summary.FileWriter(summary_path, sess.graph)

    epoch_learning_rate = init_learning_rate
    #epoch_learning_rate = tmp_learning_rate
    if isTrain:
        test_acc_old=0
        for epoch in range(1, total_epochs + 1):
        #for epoch in range(1, 2):

            if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
                epoch_learning_rate = epoch_learning_rate / 10


            pre_index = 0
            train_acc = 0.0
            train_acc_norm = 0.0
            train_acc_arcface = 0.0
            train_loss = 0.0
            train_center_loss = 0.0

            train_y_equal_0 = 0
            train_y_equal_1 = 0
            train_y_equal_2 = 0
            train_y_equal_3 = 0


            for step in range(1, iteration + 1):
                if pre_index + batch_size < dataset_size:
                    batch_x = train_x[pre_index: pre_index + batch_size]
                    batch_y = train_y[pre_index: pre_index + batch_size]
                    batch_p = train_p[pre_index: pre_index + batch_size]
                else:

                    batch_x = train_x[pre_index:]
                    batch_y = train_y[pre_index:]
                    batch_p = train_p[pre_index:]


                batch_x = data_augmentation(batch_x)

                train_feed_dict = {
                    x: batch_x,
                    label: batch_y,
                    path: batch_p,
                    learning_rate: epoch_learning_rate,
                    training_flag: True
                }

                _, batch_loss= sess.run([train, cost], feed_dict=train_feed_dict)
                batch_acc = accuracy.eval(feed_dict=train_feed_dict)
                '''
                logits_watch = sess.run(logits,feed_dict=train_feed_dict)
                print("logit is",logits_watch )
                print("label is ", batch_y)
                '''


                train_loss += batch_loss
                #train_center_loss += batch_center_loss
                train_acc += batch_acc

                pre_index += batch_size

                if step == iteration:
                    train_loss /= iteration  # average loss
                    train_acc /= iteration  # average accuracy
                    train_center_loss /= iteration
                    if epoch >= total_epochs-1:
                        train_acc_norm /= iteration
                        train_acc_arcface /= iteration

                    train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                                      tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])


                    test_acc, test_loss,  test_summary= Evaluate(sess, epoch)



                    summary_writer.add_summary(summary=train_summary, global_step=epoch)
                    summary_writer.add_summary(summary=test_summary, global_step=epoch)
                    summary_writer.flush()


                    line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f\n" % (
                        epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)
                    print(line)




                    with open(txt_path, 'a') as f :
                        f.write(line)
                    if epoch >= total_epochs-10:
                        test_acc_new = test_acc
                        if test_acc_new >= test_acc_old:
                            saver_2.save(sess=sess, save_path=save_path)
                            print("model saved ,acc is", test_acc_new)
                            test_acc_old = test_acc

                if epoch >= total_epochs-1:
                    train_result = sess.run(tf.argmax(logits, 1),feed_dict=train_feed_dict)
                    label_argmax =  sess.run(tf.argmax(label, 1), feed_dict=train_feed_dict)

                    for itrain in range(len(batch_y)):
                        if label_argmax[itrain] == 0:
                            if train_result[itrain] == label_argmax[itrain]:
                                train_y_equal_0 = train_y_equal_0 + 1
                        elif label_argmax[itrain] == 1:
                            if train_result[itrain] == label_argmax[itrain]:
                                train_y_equal_1 = train_y_equal_1 + 1
                        elif label_argmax[itrain] == 2:
                            if train_result[itrain] == label_argmax[itrain]:
                                train_y_equal_2 = train_y_equal_2 + 1
                        elif label_argmax[itrain] == 3:
                            if train_result[itrain] == label_argmax[itrain]:
                                train_y_equal_3 = train_y_equal_3 + 1
                    #s_train_val = sess.run(s_train, feed_dict=train_feed_dict)

            if epoch >= total_epochs-1:
                #print("s_train_val is ", s_train_val)
                print("the right amount of train of 0 and 1 and 2 and 3 is  ", train_y_equal_0, train_y_equal_1, train_y_equal_2, train_y_equal_3)



    else:
        epoch = total_epochs-1

        test_acc, test_loss, test_summary = Evaluate(sess, epoch)
        print("test_loss:",test_loss,"test_acc",test_acc)

end_time = time.time()
print("end time is", end_time)

time_dur = end_time - start_time
print("time_dur is ", time_dur)
