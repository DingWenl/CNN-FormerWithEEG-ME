import keras
from keras.models import load_model
import scipy.io as scio 
import random
import numpy as np
from scipy import signal
import os
from random import sample
from keras.utils import np_utils
import pandas as pd
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

# get the training sampels
def datagenerator(batchsize,train_data1,train_data2,train_data3,win_train,channel,test_list):
    x_train1, x_train2, x_train3, y_train = list(range(batchsize)), list(range(batchsize)), list(range(batchsize)), list(range(batchsize))
    target_list = list(range(40))
    for i in range(batchsize):
        k = test_list[0]
        m = sample(target_list, 1)[0]
        time_start = random.randint(int(35+125),int(1250+35+125-win_train))
        time_end = time_start + win_train
        x_11 = train_data1[k][m][:,time_start:time_end]
        x_21 = np.reshape(x_11,(channel, win_train, 1))
        x_train1[i]=x_21
        
        x_12 = train_data2[k][m][:,time_start:time_end]
        x_22 = np.reshape(x_12,(channel, win_train, 1))
        x_train2[i]=x_22

        x_13 = train_data3[k][m][:,time_start:time_end]
        x_23 = np.reshape(x_13,(channel, win_train, 1))
        x_train3[i]=x_23

        y_train[i] = np_utils.to_categorical(m, num_classes=40, dtype='float32')
        
    x_train1 = np.reshape(x_train1,(batchsize,channel, win_train, 1))
    x_train2 = np.reshape(x_train2,(batchsize,channel, win_train, 1))
    x_train3 = np.reshape(x_train3,(batchsize,channel, win_train, 1))

    x_train = np.concatenate((x_train1, x_train2, x_train3), axis=-1)
    y_train = np.reshape(y_train,(batchsize,40))
    
    return x_train, y_train

# get the filtered EEG-data, more details refer to the "get_train_data" in "train.py"
def get_test_data(wn11,wn21,wn12,wn22,wn13,wn23,path):
    # read the data
    data = scio.loadmat(path)
    data_1 = data['data']
    c1 = [47,53,54,55,56,57,60,61,62]
    
    train_data = data_1[c1,:,:,:]
    block_data_list1 = []
    for i in range(train_data.shape[3]):
        target_data_list = []
        for j in range(train_data.shape[2]):
            channel_data_list = []
            for k in range(train_data.shape[0]):
                b, a = signal.butter(6, [wn11,wn21], 'bandpass')
                filtedData = signal.filtfilt(b, a, train_data[k,:,j,i])
                channel_data_list.append(filtedData)
            channel_data_list = np.array(channel_data_list)
            target_data_list.append(channel_data_list)
        block_data_list1.append(target_data_list)
        
    block_data_list2 = []
    for i in range(train_data.shape[3]):
        target_data_list = []
        for j in range(train_data.shape[2]):
            channel_data_list = []
            for k in range(train_data.shape[0]):
                b, a = signal.butter(6, [wn12,wn22], 'bandpass')
                filtedData = signal.filtfilt(b, a, train_data[k,:,j,i])
                channel_data_list.append(filtedData)
            channel_data_list = np.array(channel_data_list)
            target_data_list.append(channel_data_list)
        block_data_list2.append(target_data_list)

    block_data_list3 = []
    for i in range(train_data.shape[3]):
        target_data_list = []
        for j in range(train_data.shape[2]):
            channel_data_list = []
            for k in range(train_data.shape[0]):
                b, a = signal.butter(6, [wn13,wn23], 'bandpass')
                filtedData = signal.filtfilt(b, a, train_data[k,:,j,i])
                channel_data_list.append(filtedData)
            channel_data_list = np.array(channel_data_list)
            target_data_list.append(channel_data_list)
        block_data_list3.append(target_data_list) 
    return block_data_list1, block_data_list2, block_data_list3

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # Setting hyper-parameters, more details refer to "train.py"

    fs = 250
    channel = 9

    batchsize = 5000
    f_down1 = 6
    f_up1 = 50
    wn11 = 2*f_down1/fs
    wn21 = 2*f_up1/fs
    
    f_down2 = 14
    f_up2 = 50
    wn12 = 2*f_down2/fs
    wn22 = 2*f_up2/fs
    
    f_down3 = 22
    f_up3 = 50
    wn13 = 2*f_down3/fs
    wn23 = 2*f_up3/fs
    
    total_av_acc_list = []
    # the lsit of mask ratios, using 0.2 as an example
    mask_rate_list = [0.2] # [0.0,0.1,0.2,0.3,0.4,0.5]
    # the lsit of data lengths, using 1.0 s as an example
    t_test_list = [1.0] # [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    for sub_selelct in range(1, 36):
        # the path of the dataset and you need to change it for your test
        path = '/Users/dingwenlong/Desktop/dwl/USTC_PHD/data/benchamark/S%d.mat'%sub_selelct
        # get the filtered EEG-data of the test data
        data1, data2, data3 = get_test_data(wn11,wn21,wn12,wn22,wn13,wn23,path)
        av_acc_list = []
        for t_test in t_test_list:
            win_train = int(fs*t_test)
            for mask_rate in mask_rate_list:
                acc_list = []
                for block_n in range(6):
                    # the path of the CNN-Former's model and you need to change it
                    model_path = '/data/dwl/ssvep/model/benchmark/cnnformer_crossblock_test/cnnformer_%3.1fs_%d_mask%3.1f_block%d.h5'%(t_test, sub_selelct,mask_rate,block_n)
                    model = load_model(model_path)
                    print("load successed")
                    print(t_test, sub_selelct)
                    test_list = [block_n]
                    # get the filtered EEG-data of the test samples, and the number of the samples is "batchsize" 
                    x_train,y_train = datagenerator(batchsize,data1, data2, data3, win_train, channel,test_list)
                    a, b = 0, 0
                    y_pred = model.predict(np.array(x_train))
                    true, pred = [], []
                    y_true = y_train
                    # Calculating the accuracy of current time
                    for i in range (batchsize):
                        y_pred_ = np.argmax(y_pred[i])
                        pred.append(y_pred_)
                        y_true1  = np.argmax(y_train[i])
                        true.append(y_true1)
                        if y_true1 == y_pred_:
                            a += 1
                        else:
                            b+= 1
                    acc = a/(a+b)
                    print('each blcok:',acc)
                    acc_list.append(acc)

                av_acc = np.mean(acc_list)
                print('block average:', av_acc)
                av_acc_list.append(av_acc)
        total_av_acc_list.append(av_acc_list)
    
    # save the results
    # print(total_av_acc_list)
    company_name_list = total_av_acc_list
    df = pd.DataFrame(company_name_list)
    df.to_excel("/home/Dwl/result/benchmark.xlsx", index=False)
    
    