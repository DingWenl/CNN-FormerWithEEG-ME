from keras.callbacks import ModelCheckpoint
from net_cnnformer import cnnformer
import data_generator
import scipy.io as scio 
from scipy import signal
from keras.models import Model
from keras.layers import Input
import numpy as np
import os

# get the filtered EEG-data, label and the start time of each trial of the dataset
def get_train_data(wn11,wn21,wn12,wn22,wn13,wn23,path):
    # read the data
    data = scio.loadmat(path)
    data_1 = data['data']
    # get the EEG data of selected 9 electrodes
    c1 = [47,53,54,55,56,57,60,61,62]
    
    train_data = data_1[c1,:,:,:]
    # get the filtered EEG-data with six-order Butterworth filter of the first sub-filter
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
    # get the filtered EEG-data with six-order Butterworth filter of the second sub-filter
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
    # get the filtered EEG-data with six-order Butterworth filter of the third sub-filter
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
    # open the GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    #%% Setting hyper-parameters
    # ampling frequency after downsampling
    fs = 250
    # the number of the electrode channels
    channel = 9
    # the hyper-parameters of the training process
    train_epoch = 400
    batchsize = 256
    
    # the filter ranges of the four sub-filters in the filter bank
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
    
    # the lsit of mask ratios, using 0.2 as an example
    mask_rate_list = [0.2] # [0.0,0.1,0.2,0.3,0.4,0.5]
    # the lsit of data lengths, using 1.0 s as an example
    t_train_list = [1.0] # [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    #%% Training the models of multi-subjects
    # selecting the training subject
    for sub_selelct in range(1, 36):
        # the path of the dataset and you need change it for your training
        path = '/data/dwl/ssvep/benchmark/S%d.mat'%sub_selelct
        # get the filtered EEG-data of three sub-input of the training data
        data1, data2, data3 = get_train_data(wn11,wn21,wn12,wn22,wn13,wn23,path)
        # selecting the training time-window
        for t_train in t_train_list:
            # transfer time to frame
            win_train = int(fs*t_train)
            for mask_rate in mask_rate_list:
            # the traing data is randomly divided in the traning dataset and validation set according to the radio of 9:1
                for block_n in range(6):
                    train_list = list(range(6))
                    val_list = [block_n]
                    train_list = [i for i in train_list if (i not in val_list)]
                    # data generator (generate the taining and validation samples of batchsize trials)
                    train_gen = data_generator.train_datagenerator(batchsize,data1, data2, data3,win_train,train_list, channel,mask_rate)#, t_train)
                    #%% setting the input of the network
                    input_shape = (channel, win_train, 3)
                    input_tensor = Input(shape=input_shape)
                    # using the CNN-Former model
                    preds = cnnformer(input_tensor)
                    model = Model(input_tensor, preds)
                    # the path of the saved model and you need to change it
                    model_path = '/data/dwl/ssvep/model/benchmark/cnnformer_crossblock_test/cnnformer_%3.1fs_%d_mask%3.1f_block%d.h5'%(t_train, sub_selelct,mask_rate,block_n)
                    # some hyper-parameters in the training process
                    model_checkpoint = ModelCheckpoint(model_path, monitor='loss',verbose=1, save_best_only=True,mode='auto')
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    # training, using model.fit or model.fit_generator
                    history = model.fit(
                            train_gen,
                            steps_per_epoch= 10,
                            epochs=train_epoch,
                            validation_data=None,
                            validation_steps=1,
                            callbacks=[model_checkpoint]
                            )
    
    
    
    
    
    # # show the process of the taining
    # epochs=range(len(history.history['loss']))
    # plt.subplot(221)
    # plt.plot(epochs,history.history['accuracy'],'b',label='Training acc')
    # plt.plot(epochs,history.history['val_accuracy'],'r',label='Validation acc')
    # plt.title('Traing and Validation accuracy')
    # plt.legend()
    # # plt.savefig('D:/dwl/code_ssvep/DL/cross_session/m_coyy/photo/model_V3.1_acc1.jpg')
    
    # plt.subplot(222)
    # plt.plot(epochs,history.history['loss'],'b',label='Training loss')
    # plt.plot(epochs,history.history['val_loss'],'r',label='Validation val_loss')
    # plt.title('Traing and Validation loss')
    # plt.legend()
    # # plt.savefig('D:/dwl/code_ssvep/DL/cross_session/m_coyy/photo/model_2.5s_loss0%d.jpg'%sub_selelct)
    
    # plt.show()





