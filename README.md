# CNN-Former model with EEG-ME for SSVEP classification
Here are the codes of the CNN-Former with EEG-ME in the paper "A Novel Data Augmentation Approach Using Mask Encoding for Deep Learning-Based Asynchronous SSVEP-BCI“，which is accepted by the IEEE Transactions on Neural Systems & Rehabilitation Engineering.
## The core code of EEG-ME in data_generator.py
1. segment = list(range(int(win_train-mask_rate*win_train)))
2. r_m = sample(segment, 1)[0]
3. x_train[:,:,r_m:int(r_m+mask_rate*win_train),:] = 0
4. where "r_m" denotes the start point of the mask window, "win_train" denotes the data length of the time window of training  samples, "mask_rate" denotes the mask ratio of the training samples, and "x_train" denotes the mini-batch training samples 


## The related version information
1. Python == 3.7.0
2. Keras-gpu == 2.3.1
3. tensorflow-gpu == 2.1.0
4. scipy == 1.5.2
5. numpy == 1.19.2
## Training CNN-Former with EEG-ME for the benchmark dataset
1. Download the code.
2. Download the [benchmark dataset](http://bci.med.tsinghua.edu.cn/download.html) and its [paper](https://ieeexplore.ieee.org/abstract/document/7740878).
3. Create a model folder to save the model.
4. Change the data and model folder paths in train and test files to your data and model folder paths.

