# -*- coding:utf-8 -*-

import os
import sys
import time
import nibabel as nib
import tensorlayer as tl
import pickle  #用于python特有的类型和python的数据类型间进行转换  序列化 反序化 存储传输有关
import random
import numpy as np


class_num = 4
class_num_total = 4
image_size = 64
img_channels = 1


# ========================================================== #
# ├─ prepare_data()
#  ├─ download training data if not exist by download_data()
#  ├─ load data by load_data()
#  └─ shuffe and return data
# ========================================================== #



def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data_one(file):
    batch = unpickle(file)
    data = batch['data']
    labels = batch['labels']
    path = batch['path']
    print("Loading %s : %d." % (file, len(data)))
    return data, labels, path


def load_data(files, data_dir, label_count):
    global image_size, img_channels
    data, labels, path = load_data_one(data_dir + '/' + files)
    labels = np.array([[float(i == label-1) for i in range(label_count)] for label in labels])
    #print("the label is ", labels)
    print("label_count is ", label_count)
    data = data.reshape([-1, img_channels, image_size, image_size])
    data = data.transpose([0, 2, 3, 1])
    return data, labels, path


def prepare_data(train_files, test_files):
    print("======Loading data======")
    data_dir = '/data/wangshuai/shuju/dataset'
    #meta = unpickle(data_dir + '/training.pickle')


    label_count = class_num
    #train_files = '/training_64_1_234.pickle'
    #test_files = '/test_64_1_234.pickle'
    train_data, train_labels, train_path = load_data(train_files, data_dir, label_count)
    test_data, test_labels, test_path = load_data(test_files, data_dir, label_count)
    print("Now the shape of label is " ,train_labels.shape,test_labels.shape )

    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    print("======Shuffling data======")
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    train_path = np.array(train_path)[indices]

    print("======Prepare Finished======")
    return train_data, train_labels, train_path, test_data, test_labels, test_path



# ========================================================== #
# ├─ _random_crop()
# ├─ _random_flip_leftright()
# ├─ data_augmentation()
# └─ color_preprocessing()
# ========================================================== #

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])
    #print("oshape is ", oshape)

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    #print("oshpe new is ", oshape)

    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    #x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    #x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    #x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    #x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])

    return x_train, x_test


def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [64, 64], 4)
    #print("the shape of batch is ",np.shape(batch))
    return batch