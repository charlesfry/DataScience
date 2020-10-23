import ast
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
from math import ceil
import numpy as np
import pandas as pd
import wfdb


def seed_everything(seed=0) :
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    try :
        tf.random.set_seed(seed)
    except :
        return

seed = 69
seed_everything(seed)

def load_raw_data(df,sampling_rate,path) :
    if sampling_rate == 100 :
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else :
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal,meta in data])
    return data

path = 'E:\DataScience\ptb_xl\\'
sampling_rate = 100

# load and convert annotation data
Y = pd.read_csv(path + 'ptbxl_database.csv',index_col='ecg_id')

Y.scp_codes = Y.scp_codes.apply(
    lambda x: ast.literal_eval(x)
)


# load raw signal data
X = load_raw_data(df=Y,sampling_rate=sampling_rate,path=path)

# load scp_statements.csv ofr diagnostic aggregation
agg_df = pd.read_csv(path + 'scp_statements.csv',index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic:pd.DataFrame) :
    tmp = []
    for key in y_dic.keys() :
        if key in agg_df.index :
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

def to_one_hot(Y) :
    Y = pd.DataFrame(mlb.fit_transform(Y),
                            columns=mlb.classes_)
    return Y

# split into train/test
dev_fold = 9
test_fold = 10

# train
X_train = X[np.where(Y.strat_fold < dev_fold)]
Y_train = Y[Y.strat_fold < dev_fold].diagnostic_superclass

# dev
X_dev = X[np.where(Y.strat_fold == dev_fold)]
Y_dev = Y[Y.strat_fold == dev_fold].diagnostic_superclass

# test
X_test = X[np.where(Y.strat_fold == test_fold)]
Y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

Y_train = to_one_hot(Y_train)
Y_dev = to_one_hot(Y_dev)
Y_test = to_one_hot(Y_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Input,BatchNormalization,Flatten
from tensorflow.keras.regularizers import L1
from tensorflow_addons.layers import WeightNormalization




def build_model() :

    model = Sequential([
        Input(batch_input_shape=(None, 1000, 12)),

        BatchNormalization(),
        WeightNormalization(Dense(512, activation='relu', kernel_initializer='he_normal')),
        Dropout(.6),

        BatchNormalization(),
        WeightNormalization(Dense(512, activation='relu', kernel_initializer='he_normal')),
        Dropout(.4),

        BatchNormalization(),
        WeightNormalization(Dense(256, activation='relu', kernel_initializer='he_normal')),
        Dropout(.2),

        BatchNormalization(),
        WeightNormalization(Dense(128, activation='relu', kernel_initializer='he_normal')),
        Dropout(.2),

        Flatten(),

        BatchNormalization(),
        Dense(5,activation='sigmoid', kernel_initializer='glorot_normal')
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )



    return model



# batch_size = ceil(X_train.size / 2)

# model.fit(X_train,Y_train,epochs=50,callbacks=callbacks,batch_size=batch_size,steps_per_epoch=2)

modelllllll = build_model()

# evaluate_model(model)

from tensorflow.keras.layers import Conv1D,MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer,Dense,Dropout,\
    Activation,Flatten,Reshape,Permute
from tensorflow.keras.layers import MaxPooling1D,UpSampling1D,Cropping1D

from tensorflow.keras.layers import Conv1DTranspose

from tensorflow.keras import backend as kb
l2 = tf.keras.regularizers.l2(0.0001)

def build_cnn() :
    model = Sequential([
        Input(batch_input_shape=(None, 1000, 12)),

        BatchNormalization(axis=1,gamma_regularizer=l2,beta_regularizer=l2),
        Activation('relu'),
        Conv1D(32,kernel_size=(3),input_shape=(1000,12),kernel_initializer='he_uniform'),
        MaxPooling1D((2)),
        Dropout(.2),

        BatchNormalization(axis=1,gamma_regularizer=l2,beta_regularizer=l2),
        Activation('relu'),
        Conv1D(64,(3),kernel_initializer='he_uniform'),
        MaxPooling1D((2)),
        Dropout(.2),

        Activation('relu'),
        Conv1D(128,(3)),

        Flatten(),
        BatchNormalization(gamma_regularizer=l2,beta_regularizer=l2),
        Dense(5, activation='sigmoid', kernel_initializer='glorot_normal')
        # Dense(1, activation='sigmoid', kernel_initializer='glorot_normal')
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )


    return model



cnn_model = build_cnn()

from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,TensorBoard,ModelCheckpoint,CSVLogger


# ----- Model ----- #
kernel_size = 16
kernel_initializer = 'he_normal'
batch_size = 120
lr = 0.001

callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=7,
                                   min_lr=lr / 100),
                 EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
                               min_delta=0.00001)]
callbacks += [TensorBoard(log_dir='./logs', batch_size=batch_size, write_graph=False),
                  CSVLogger('training.log', append=False)]  # Change append to true if continuing training
# Save the BEST and LAST model
callbacks += [ModelCheckpoint('./backup_model_last'),
                ModelCheckpoint('./backup_model_best', save_best_only=True)]

def find_batch_size(model) :
    for i in range(X_train.shape[0] // 10,1,-X_train.shape[0] // 50) :
        print(f'Fitting model with batch size {i}')
        try:
            model.fit(X_train, Y_train, epochs=1, batch_size=i)
            print(f'Success with batch size {i}')
            return i
        except:
            print('\tOOM. Trying with lower batch size')

# ----------------- #



if __name__ == "__main__":

    batch_size = 700

    cnn_model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  epochs=70,
                  shuffle='batch',
                  initial_epoch=0,  # If you are continuing an interrupted section change here
                  callbacks=callbacks,
                  validation_data=(X_dev, Y_dev),
                  verbose=1)
    cnn_model.evaluate(X_test, Y_test)

    from xgboost import XGBClassifier
    from sklearn.metrics import log_loss

    for i in range(Y_train.shape[1]) :
        y_train = Y_train[:,i]
        y_dev = Y_dev[:,i]
        xgb = XGBClassifier()
        xgb.fit(X_train,y_train)
        print(f'\nXGBClassifier performance on column {i}:')
        print(log_loss(y_dev,xgb.predict_proba(X_dev)))