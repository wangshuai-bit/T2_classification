# add "dicom read.py"
# add the rectangle on the image 20190411
# read the file for batch 20190413
# add function for findPosition and findFolder
# scale the image selected to 32*32
# correct the label,now csdn6 is the best for dataset
# reshape the image to vector 20180417,the label is one number
# with del 20190421
# 2 kinds of data 20190429
# 2 kinds of folders,add T2WI
# img_dicom_cropped into 0-255 20190522
# select the ROI only, not all
# add the function histequ 20190528
# cut the image to 128 20190617
# cut the image to 64 20190624
# add the expand list
# 5class 20190711
# move function dicomFIleName out of the "for" 20190711
# delete gauss in training 20190830
# add the flip in training data

import pickle
import tensorlayer as tl
import numpy as np
import os
import nibabel as nib
import dicom
import os.path
from PIL import Image
import random
from skimage import transform,data
import cv2 as cv

import json

import threading
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy
from tensorlayer.prepro import *
import skimage.measure

nib.Nifti1Header.quaternion_threshold = - np.finfo(np.float32).eps * 10  # 松弛一下限制
#root_path = "/data/wangshuai/shuju"
root_path = "/data/wangshuai/shuju/single"

flag="single"

#training_data_path = "/data/wangshuai/data_test/T2FS"
preserving_ratio = 0.000025 # filter out 2d images containing < 25% non-zeros

X_train = []
X_label = []
x_path = []
expand_list = [0,0,0,0,0,0,0,0,0,0]
X_label_cnt = [0,0,0,0,0]
X_slice_cnt = [0,0,0,0,0]
datasets = {}




def listDir(rootDir):
    #find the path
    training_data_path = []
    training_data_property = []
    training_data_folder_property = []
    training_data_filename = []
    for data_property in os.listdir(rootDir):
        data_path_pre = os.path.join(root_path,data_property)
        for filename in os.listdir(data_path_pre):
            data_path = os.path.join(data_path_pre,filename,"resave")
            data_path_try = os.path.join(data_path,"I1000000")
            if os.path.exists(data_path_try):
                folder_property = "T2FS"
            else:
                folder_property = "T2WI"
            training_data_path.append(data_path)
            training_data_property.append(data_property)
            training_data_folder_property.append(folder_property)
            training_data_filename.append(filename)
    print("the path of the nii is ", training_data_path,"and the property is ", training_data_property,"the training_data_folder_property is",training_data_folder_property )
    print ("-------------------------------------------------------function listDir finished!" )
    return training_data_path, training_data_property, training_data_folder_property, training_data_filename

training_data_path, training_data_property, training_data_folder_property, training_data_filename = listDir(root_path)
#print("-------------------------------------------------------------------------------")


def loadFileInformation(filename):
    #read the nii

    information = {}
    ds = dicom.read_file(filename)
    information['w'] = ds.Rows
    information['h'] = ds.Columns
    information['type'] = ds.Modality
    information['PatientBirthDate'] = ds.PatientBirthDate
    information['PatientSex'] = ds.PatientSex
    information['StudyID'] = ds.StudyID
    information['StudyDate'] = ds.StudyDate
    information['StudyTime'] = ds.StudyTime
    information['SOP Instance UID '] = ds.SOPInstanceUID
    information['Manufacturer'] = ds.Manufacturer

    InstanceNumber = ds.InstanceNumber
    pix = ds.pixel_array
    pix_arr = np.array(pix, dtype =  np.int)
    #print('the type of pix is ',type(pix) )
    print("the instance number is ",InstanceNumber)
    #print("the type of pix_arr is ",type(pix_arr))
    #print("the shape of pix_arr is ",pix_arr.shape)

    return  InstanceNumber, pix_arr


def findPosition(img_2d):
    # find the position in the nii
    position = np.where(img_2d)
    img_left_index = min(position[0])
    img_right_index = max(position[0])
    img_up_index = min(position[1])
    img_down_index = max(position[1])
    center_x = int((img_left_index + img_right_index) / 2)
    center_y = int((img_up_index + img_down_index) / 2)

    print("the left,right,up,down,center_x,center_y is ", img_left_index, img_right_index, img_up_index, img_down_index,
          center_x, center_y)
    print("the nozero is", np.count_nonzero(img_2d))

    for i in range(0, 100):
        print("i is ", i)

        left_cropped_pos = center_x - 32 * i
        right_cropped_pos = center_x + 32 * i
        up_cropped_pos = center_y - 32 * i
        down_cropped_pos = center_y + 32 * i

        left_done = bool(max(img_2d[left_cropped_pos, :]) == 0)
        right_done = bool(max(img_2d[right_cropped_pos, :]) == 0)
        up_done = bool(max(img_2d[:, up_cropped_pos]) == 0)
        down_done = bool(max(img_2d[:, down_cropped_pos]) == 0)

        if (left_done and right_done and up_done and down_done):
            expand = i
            break
        else:
            continue

    expand_list[expand] = expand_list[expand] + 1
    print("expand_coef is", expand)
    print("left,right,up,down_cropped_pos is", left_cropped_pos, right_cropped_pos, up_cropped_pos, down_cropped_pos)
    print("----------function findPosition finished!")

    return img_left_index, img_right_index, img_up_index, img_down_index, center_x, center_y, expand, left_cropped_pos, \
           right_cropped_pos, up_cropped_pos, down_cropped_pos



def findFolder(training_data_path_i, training_data_folder_property_i):
    #find the dicom folder
    training_data_path_i_list = os.listdir(training_data_path_i)
    #print("training_data_path_i_list is ", training_data_path_i_list)

    dir_name = training_data_path_i

    print("dir_name", dir_name)
    print("-------------------------------------------function findFolder finished!")
    return dir_name

def dicomFIleName(training_data_folder_property_i):
    #this is a list for dicom
    if training_data_folder_property_i == "T2WI":
        dicom_name = [
        "IM0",
        "IM1",
        "IM2",
        "IM3",
        "IM4",
        "IM5",
        "IM6",
        "IM7",
        "IM8",
        "IM9",
        "IM10",
        "IM11",
        "IM12",
        "IM13",
        "IM14",
        "IM15",
        "IM16",
        "IM17",
        "IM18",
        "IM19",
        "IM20",
        "IM21",
        "IM22",
        "IM23",
        "IM24",
        "IM25",
        "IM26",
        "IM27",
        "IM28",
        "IM29",
        "IM30"
        ]
    else:
        dicom_name = [
        "I1000000",
        "I2000000",
        "I3000000",
        "I4000000",
        "I5000000",
        "I6000000",
        "I7000000",
        "I8000000",
        "I9000000",
        "I1000001",
        "I1100000",
        "I1200000",
        "I1300000",
        "I1400000",
        "I1500000",
        "I1600000",
        "I1700000",
        "I1800000",
        "I1900000",
        "I2000001",
        "I2100000",
        "I2200000",
        "I2300000",
        "I2400000",
        "I2500000",
        "I2600000",
        "I2700000",
        "I2800000",
        "I2900000",
        "I3000001"
        ]
    print("type of dicom_name is", type(dicom_name))
    print(dicom_name)
    print("--------------------------------------function dicomFIleName finished!")
    return dicom_name

def histequ(grey, nlevels = 256):
    # compute histogram
    histogram = np.bincount(grey.flatten(), minlength=nlevels)
    #print("histogram:", histogram)

    #mapping function
    uniform_hist = (nlevels-1)*(np.cumsum(histogram))/(grey.size*1.0)
    uniform_hist = uniform_hist.astype('uint8')
    #print("uniform hist",uniform_hist)

    hight, width = grey.shape
    uniform_grey = np.zeros(grey.shape,dtype='uint8')
    for i in range(hight):
        for j in range(width):
            uniform_grey[i,j] = uniform_hist[grey[i,j]]
    print("------------------uniform grey finished----------------------- ")
    return uniform_grey

def rot(x):
    y = np.rot90(x)
    return y




print("---------------------now we will read the image-----------------------")

for i, val in enumerate(training_data_path): # val is the element in training data path  ['/data/wangshuai/shuju/total/2/1485699/resave',...]
    print("now processing this path : number:{} , value:{}".format(i, training_data_path[i]))
    training_data_path_i = training_data_path[i]
    training_data_property_i = training_data_property[i]
    training_data_folder_property_i = training_data_folder_property[i]
    training_data_filename_i = training_data_filename[i]
    # read the nii
    f_train = tl.files.load_file_list(path=training_data_path_i,
                                      regx='.*.gz',
                                      printable=False)  # 将test测试集合中的数据以list形式存下来 返回的是文件列表
    train_3d_num = len(f_train)
    print("the nii data  is ", f_train)

    #find the dicom folder
    dicom_folder = findFolder(training_data_path_i, training_data_folder_property_i)
    print("printplot_cnt is ", i)
    if i <= 10:
        printplot = True
        printplot_cnt = i
    for fi, f in enumerate(f_train):   # 相当于取出下标索引以及list里面相关的数据
        patch_num = 0
        print("processing fi/train_3d_num [{}/{}] 3d image f({}) for training set ...".format(fi + 1, train_3d_num, f))

        #the path of nii
        img_path = os.path.join(training_data_path_i, f)
        print("the path of the nii image is:",img_path)

        #the data of nii
        img = nib.load(img_path).get_data()
        print("the shape of the image is:",img.shape)
        img_3d_max = np.amax(img)
        img = img / img_3d_max * 1  # 对所求的像素进行归一化变成0-1范围,这里就是三维数据
        X_dicom_name = dicomFIleName(training_data_folder_property_i) # move out of the "for" 20190711

        for i in range(img.shape[2]):   # 对切片进行循环
            # read the dicom

            dicom_path = os.path.join(dicom_folder, X_dicom_name[i])
            X_dicom_name_i = X_dicom_name[i]

            X_dicom_number, X_dicom_pix_arr = loadFileInformation(dicom_path)
            print("read finished,the i is, and the path is, and the number is ,and the nii number is ", i, dicom_path, X_dicom_number, img.shape[2] - X_dicom_number)

            img_2d = img[:, :, img.shape[2] - X_dicom_number]  # 取出一张图像  img_2d is the nii_data
            #print("the shape of img_2d nii is ", img_2d.shape)
            if float(np.count_nonzero(img_2d)) / img_2d.size >= preserving_ratio:  # 表示一副图像非0个数超过整副图像的10%我们才把该图像保留下来
                #img_2d = img_2d / 127.5 - 1  # 对最初的0-255图像进行归一化到[-1, 1]范围之内
                img_2d = np.transpose(img_2d, (1, 0))  # 这个相当于将图像进行旋转90度


                # fig1 = plt.figure("dicom")
                # plt.imshow(X_dicom_pix_arr)
                # plt.show()

                #find the position of nii
                img_left_index, img_right_index, img_up_index, img_down_index, center_x, center_y, expand, left_cropped_pos, \
                right_cropped_pos, up_cropped_pos, down_cropped_pos = findPosition(img_2d)


                #crop the image
                img_selected = img_2d[img_left_index:img_right_index, img_up_index:img_down_index]
                print("the shape of image img_selected is ", img_selected.shape)

                img_cropped = img_2d[left_cropped_pos:right_cropped_pos, up_cropped_pos:down_cropped_pos]
                #img_dicom_cropped = X_dicom_pix_arr[left_cropped_pos:right_cropped_pos, up_cropped_pos:down_cropped_pos]
                img_dicom_muti = X_dicom_pix_arr*img_2d
                img_dicom_cropped = img_dicom_muti[left_cropped_pos:right_cropped_pos, up_cropped_pos:down_cropped_pos]
                #img_dicom_cropped into 0-255  20190522
                img_dicom_cropped_max = np.amax(img_dicom_cropped)
                img_dicom_cropped = img_dicom_cropped / img_dicom_cropped_max * 255
                img_dicom_cropped = img_dicom_cropped.astype('uint8')
                print("the shepe of img_dicom_cropped", img_dicom_cropped.shape)
                #print("img_dicom_cropped is ", img_dicom_cropped)
                img_dicom_cropped_pre = img_dicom_cropped
                img_dicom_cropped = histequ(img_dicom_cropped)

                # label
                if training_data_property_i == '1':
                    X_label_i = 1
                elif training_data_property_i == '2':
                    X_label_i = 2
                elif training_data_property_i == '3':
                    X_label_i = 3
                elif training_data_property_i == '4':
                    X_label_i = 4
                elif training_data_property_i == '5':
                    X_label_i = 5

                X_slice_cnt[X_label_i-1] +=1
                #img_dicom_cropped_angle_90 = np.rot90(img_dicom_cropped)
                #img_dicom_cropped_angle_180 = np.rot90(img_dicom_cropped_angle_90)

                img_dicom_cropped_norm = []
                X_label_norm = []
                training_data_path_norm = []


                for i in range (0,expand):
                    for j in range (0,expand):
                        img_dicom_cropped_norm_i = img_dicom_cropped[64*i:64*i+64, 64*j:64*j+64]
                        img_dicom_cropped_norm.append(img_dicom_cropped_norm_i)
                        X_label_norm.append(X_label_i)
                        training_data_path_norm.append(training_data_filename_i)
                        X_label_cnt[X_label_i - 1] += 1
                        print("for X_label_i is",X_label_i)
                        patch_num += 1
                '''
                if flag=="train" :
                    for i in range (0,expand):
                        for j in range (0,expand):
                            img_dicom_cropped_norm_i = img_dicom_cropped_pre[64*i:64*i+64, 64*j:64*j+64]
                            img_dicom_cropped_norm.append(img_dicom_cropped_norm_i)
                            X_label_norm.append(X_label_i)
                            training_data_path_norm.append(training_data_filename_i)
                            X_label_cnt[X_label_i - 1] += 1
                            print("for X_label_i is",X_label_i)
                
                if expand <=2:
                    img_dicom_cropped_norm.append(img_dicom_cropped)
                    X_label_norm.append(X_label_i)
                    training_data_path_norm.append(training_data_filename_i)
                    X_label_cnt[X_label_i-1] += 1  #tell me the total number of all kinds of data
                else:
                    for i in range (0,expand-1):
                        for j in range (0,expand-1):
                            img_dicom_cropped_norm_i = img_dicom_cropped[64*i:64*i+64, 64*j:64*j+64]
                            img_dicom_cropped_norm.append(img_dicom_cropped_norm_i)
                            X_label_norm.append(X_label_i)
                            training_data_path_norm.append(training_data_filename_i)
                            X_label_cnt[X_label_i - 1] += 1  #tell me the total number of all kinds of data
                '''


                print("for X_label_i is",X_label_i)

                print("the expand and the length of img_dicom_cropped_norm is :", expand, len(img_dicom_cropped_norm))
                print("the expand and the length of X_label_norm is :", expand, len(X_label_norm))




                if expand >= 10:
                #if expand >= 7:
                    print("now there is a picture for test")
                    fig2 = plt.figure("picture")
                    plt.subplot(2,3,1)
                    rect = plt.Rectangle((up_cropped_pos,left_cropped_pos), right_cropped_pos - left_cropped_pos,
                                         down_cropped_pos - up_cropped_pos,fill = False, edgecolor="red")
                    plt.gca().add_patch(rect)
                    plt.imshow(img_2d)
                    plt.subplot(2,3,2)
                    plt.imshow(img_selected)
                    plt.subplot(2,3,3)
                    plt.imshow(img_cropped)

                    plt.subplot(2,3,4)
                    plt.title(X_dicom_name_i)
                    rect = plt.Rectangle((up_cropped_pos,left_cropped_pos), right_cropped_pos - left_cropped_pos,
                                         down_cropped_pos - up_cropped_pos,fill = False, edgecolor="red")
                    plt.gca().add_patch(rect)
                    plt.imshow(X_dicom_pix_arr)

                    plt.subplot(2,3,5)
                    plt.title(printplot_cnt)
                    plt.imshow(img_dicom_cropped)
                    plt.subplot(2,3,6)
                    plt.imshow(img_dicom_cropped_pre)
                    plt.pause(0.01)


                    fig4 = plt.figure("img_dicom_cropped_norm")
                    for i in range(0,len(img_dicom_cropped_norm)):
                        plt.subplot((expand-1)*2, (expand-1)*2, i+1)
                        plt.imshow(img_dicom_cropped_norm[i])
                    plt.pause(0.01)




                X_train = X_train + img_dicom_cropped_norm

                print("the langth of X_train is :",len(X_train))

                #append the label into list X_label_norm
                X_label = X_label + X_label_norm
                print("the length of X_label is :", len(X_label))
                x_path = x_path + training_data_path_norm
                print("------------------------------------------this picture process finished")
        del img
        del X_label_norm
        del img_dicom_cropped_norm
        del X_dicom_pix_arr
        del img_dicom_cropped
        del img_cropped
        del img_selected
        print("in the path ", img_path, "there are ",patch_num,"patches" )


    print("------------------------------------------this folder process finished!")
print("the expand list is ", expand_list)
print("---------------------------------------the dataset read finished,the length of the list is", len(X_train))

print("the cnt of all kinds of data is ",X_label_cnt)
print("the cnt of slice of all kinds of data is ", X_slice_cnt)
print(len(X_train))

# assign the train and test
# class 1
train_data_1 = []
train_label_1 = []
test_data_1 = []
test_label_1 = []
train_path_1 = []
test_path_1 = []

train_data_2 = []
train_label_2 = []
test_data_2 = []
test_label_2 = []
train_path_2 = []
test_path_2 = []

train_data_3 = []
train_label_3 = []
test_data_3 = []
test_label_3 = []
train_path_3 = []
test_path_3 = []

train_data_4 = []
train_label_4 = []
test_data_4 = []
test_label_4 = []
train_path_4 = []
test_path_4 = []


train_data = []
train_label = []
train_path = []
test_data = []
test_label = []
test_path = []


X_label_1_cnt = 0
X_label_2_cnt = 0
X_label_3_cnt = 0
X_label_4_cnt = 0
X_label_5_cnt = 0

for i in range (len(X_train)):
    if X_label[i] == 1:
        X_label_1_cnt = X_label_1_cnt + 1
        train_data_1.append(X_train[i])
        train_label_1.append(X_label[i])
        train_path_1.append(x_path[i])
    elif X_label[i] == 2:
        X_label_2_cnt = X_label_2_cnt + 1
        train_data_2.append(X_train[i])
        train_label_2.append(X_label[i])
        train_path_2.append(x_path[i])
    elif X_label[i] == 3:
        X_label_3_cnt = X_label_3_cnt + 1
        train_data_3.append(X_train[i])
        train_label_3.append(X_label[i])
        train_path_3.append(x_path[i])
    elif X_label[i] == 4:
        X_label_4_cnt = X_label_4_cnt + 1
        train_data_4.append(X_train[i])
        train_label_4.append(X_label[i])
        train_path_4.append(x_path[i])
    elif X_label[i] == 5:
        X_label_5_cnt = X_label_5_cnt + 1



print("the lenth of data1  and label is", len(train_data_1), len(train_label_1),len(test_data_1), len(test_label_1))
print("the lenth of data2  and label is", len(train_data_2), len(train_label_2),len(test_data_2), len(test_label_2))
print("the lenth of data3  and label is", len(train_data_3), len(train_label_3),len(test_data_3), len(test_label_3))
print("the lenth of data4  and label is", len(train_data_4), len(train_label_4),len(test_data_4), len(test_label_4))
print("the cnt of all class is ", X_label_1_cnt,X_label_2_cnt,X_label_3_cnt, X_label_4_cnt, X_label_5_cnt)

if flag=="train":
    train_data_1_rot90 = list(map(rot, train_data_1))
    train_label_1_rot90 = train_label_1
    train_path_1_rot90 = train_path_1
    train_data_1 = train_data_1 + train_data_1_rot90
    train_label_1 = train_label_1 + train_label_1_rot90
    train_path_1 = train_path_1 + train_path_1_rot90

    train_data_1_rot180 = list(map(rot, train_data_1_rot90))
    train_label_1_rot180 = train_label_1_rot90
    train_path_1_rot180 = train_path_1
    train_data_1 = train_data_1 + train_data_1_rot180
    train_label_1 = train_label_1 + train_label_1_rot180
    train_path_1 = train_path_1 + train_path_1_rot180

    train_data_2_rot90 = list(map(rot, train_data_2))
    train_label_2_rot90 = train_label_2
    train_path_2_rot90 = train_path_2
    train_data_2 = train_data_2 + train_data_2_rot90
    train_label_2 = train_label_2 + train_label_2_rot90
    train_path_2 = train_path_2 + train_path_2_rot90

    train_data_2_rot180 = list(map(rot, train_data_2_rot90))
    train_label_2_rot180 = train_label_2_rot90
    train_path_2_rot180 = train_path_2
    train_data_2 = train_data_2 + train_data_2_rot180
    train_label_2 = train_label_2 + train_label_2_rot180
    train_path_2 = train_path_2 + train_path_2_rot180

    train_data_3_rot90 = list(map(rot, train_data_3))
    train_label_3_rot90 = train_label_3
    train_path_3_rot90 = train_path_3
    train_data_3 = train_data_3 + train_data_3_rot90
    train_label_3 = train_label_3 + train_label_3_rot90
    train_path_3 = train_path_3 + train_path_3_rot90

    train_data_3_rot180 = list(map(rot, train_data_3_rot90))
    train_label_3_rot180 = train_label_3_rot90
    train_path_3_rot180 = train_path_3
    train_data_3 = train_data_3 + train_data_3_rot180
    train_label_3 = train_label_3 + train_label_3_rot180
    train_path_3 = train_path_3 + train_path_3_rot180

    train_data_4_rot90 = list(map(rot, train_data_4))
    train_label_4_rot90 = train_label_4
    train_path_4_rot90 = train_path_4
    train_data_4 = train_data_4 + train_data_4_rot90
    train_label_4 = train_label_4 + train_label_4_rot90
    train_path_4 = train_path_4 + train_path_4_rot90

    train_data_4_rot180 = list(map(rot, train_data_4_rot90))
    train_label_4_rot180 = train_label_4_rot90
    train_path_4_rot180 = train_path_4
    train_data_4 = train_data_4 + train_data_4_rot180
    train_label_4 = train_label_4 + train_label_4_rot180
    train_path_4 = train_path_4 + train_path_4_rot180


    train_data_1 = train_data_1[0:430]
    train_label_1 = train_label_1[0:430]
    train_path_1 = train_path_1[0:430]

    train_data_2 = train_data_2[0:430]
    train_label_2 = train_label_2[0:430]
    train_path_2 = train_path_2[0:430]

    train_data_3 = train_data_3[0:430]
    train_label_3 = train_label_3[0:430]
    train_path_3 = train_path_3[0:430]

    train_data_4 = train_data_4[0:430]
    train_label_4 = train_label_4[0:430]
    train_path_4 = train_path_4[0:430]

fig3 = plt.figure("angle")
plt.imshow(train_data_1[0])
plt.pause(0.01)

for i in range(len(train_data_1)):
    train_data.append(train_data_1[i])
    train_label.append(1)
    train_path.append(train_path_1[i])
for j in range(len(train_data_2)):
    train_data.append(train_data_2[j])
    train_label.append(2)
    train_path.append(train_path_2[j])
for k in range(len(train_data_3)):
    train_data.append(train_data_3[k])
    train_label.append(3)
    train_path.append(train_path_3[k])
for l in range(len(train_data_4)):
    train_data.append(train_data_4[l])
    train_label.append(4)
    train_path.append(train_path_4[l])
'''
train_data.append(train_data_4[l])
train_label.append(4)
train_path.append(train_path_4[l])
'''
pk_data = train_data
pk_label = train_label
pk_path = train_path

print("now trans into array")
#change the img into a vector
for i in range (len(pk_data)):
#print("the shape of img_dicom_cropped_norm[i] is", np.shape(img_dicom_cropped_norm[i]))
    pk_data[i] = pk_data[i].reshape(4096, order='C')
#append the img into list X_train


print("the lenth of data is ", len(pk_data))
print("the lenth of label is ", len(pk_label))

pk_data = np.asarray(pk_data, dtype=np.float32)  # 将训练的图像数据原来是list现在变成np.array格式
pk_label = np.asarray(pk_label, dtype=np.int)
#X_train = X_train[:, :, :, np.newaxis]  # 变成4维数据



''''''
datasets.setdefault('data',pk_data)
datasets.setdefault('labels',pk_label)
datasets.setdefault('path', pk_path)


# save data into pickle format
data_saving_path = '/data/wangshuai/shuju/dataset'
tl.files.exists_or_mkdir(data_saving_path)

print("save training set into pickle format")
#with open(os.path.join(data_saving_path, 'training.pickle'), 'wb') as f:
with open(os.path.join(data_saving_path, 'single_64_pt_all_sel_5_to_2'), 'wb') as f:
    pickle.dump(datasets, f, protocol=4)
print("====================================== all finished")



