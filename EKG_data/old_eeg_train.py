import os
import random
import sys

import numpy as np
import tensorflow as tf
from scipy.signal import resample
from scipy.signal import spectrogram

sys.path.append("./")

from extract_transform_load import extractor

def seed_everything(seed=0) :
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything(seed:=42)

def get_paths(seed=42) :
    """

    :return: lists of interictal and preictal paths to put in a dataset
    """

    types = ['interictal_segment', 'preictal_segment']
    interictal_paths = []
    preictal_paths = []

    for root, dirs, files in os.walk('./input'):
        for i, file in enumerate(files):
            if not 'Patient_1' in file : continue
            if not file.endswith('.mat') : continue

            path = os.path.join(root, file)
            segment = path[:-9]

            if segment.endswith(types[0]):
                interictal_paths.append(path)
                continue
            if segment.endswith(types[1]):
                preictal_paths.append(path)
                continue
            # there are test file types with no answer for a kaggle competition, so skip them


    # shuffle paths
    random.Random(seed).shuffle(interictal_paths)
    random.Random(seed).shuffle(preictal_paths)

    return interictal_paths,preictal_paths

interictal_paths,preictal_paths = get_paths(seed)

assert interictal_paths == get_paths(seed)[0]
assert preictal_paths == get_paths(seed)[1]

def load_from_paths(interictal_paths, preictal_paths, num_samples=None, starting_point=0,seed=seed) :
    """

    :param interictal_paths: list of interictal data
    :param preictal_paths: list of preictal data
    :param num_samples:
    :param num_samples: where to start taking from the list
    :param seed: random state for shuffle
    :return: pandas dataframe of electrode values and whether they are preictal
    """

    X = None
    Y = np.empty(0)

    paths = interictal_paths + preictal_paths
    random.Random(seed).shuffle(paths)

    if num_samples is None :
        num_samples = len(paths)

    # for each file, preprocess the file and add to dataset
    if starting_point >= len(interictal_paths) : return
    if starting_point + num_samples > len(paths) : num_samples = len(paths) - starting_point

    for i in range(starting_point,num_samples + starting_point) :
        path = paths[i]
        x,y = extractor(path)

        downsample = int(x.shape[1] * 500 / 5000)
        # downsample x
        downsampled_x = np.empty((x.shape[0],downsample))
        for row in range(x.shape[0]):
            downsampled_x[row, :] = resample(x[row, :], downsample)
        x = downsampled_x[14,:]

        # now create spectrogram
        lst = list(range(downsample))
        span = 2000 # 1000 ticks per second after downsample

        spectrograms = []

        row_specs = np.empty((256*x.shape[0],8))
        for j in lst[::span] :
            # every second have a spectrogram
            seconds = row[j:j+span]
            _,_1,sxx = spectrogram(seconds,fs=500,return_onesided=False)
            sxx = np.log1p(sxx)
            row_specs[i * 256 : i * 256 + 256,:] = sxx

            spectrograms = row_specs
        spectrograms = np.array(spectrograms)

        x = [spectrograms]

        if X is None : X = x
        else : X = np.append(X,x,axis=0)

        Y = np.append(Y,y)

    return X,Y

secs = 3000000/5000 # Number of seconds in signal X
downsample = int(secs*500)     # Number of samples to downsample
X,Y = load_from_paths(interictal_paths=interictal_paths, preictal_paths=preictal_paths, num_samples=20,
                     starting_point=0)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input,BatchNormalization
from tensorflow.keras.layers import Conv3D, MaxPooling3D

batch_size = 5
num_classes = 2
epochs = 100
img_rows, img_cols = 256,8

print(X.shape)
quit()
def build_model() :
    model = Sequential([
        Input(shape=(None,*X[0].shape)),
        BatchNormalization(),
        Conv3D(16, kernel_size=(3, 3, 3), activation='relu',input_shape=(15, 3840, 8))
    ])

    model.add(Conv3D(32, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.RMSprop(),
                  metrics=['accuracy'])
    return model

model = build_model()
model.fit(X,Y,epochs=10,batch_size=batch_size)
model.evaluate(X,Y)

model = build_model()
model.fit()