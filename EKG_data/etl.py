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
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.callbacks import EarlyStopping


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

    callbacks = [
        EarlyStopping(
            # Stop training when loss is no longer improving
            monitor="loss",
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=1e-3,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=2,
            verbose=1,
        )
    ]

    return model,callbacks

model,callbacks = build_model()

# batch_size = ceil(X_train.size / 2)

# model.fit(X_train,Y_train,epochs=50,callbacks=callbacks,batch_size=batch_size,steps_per_epoch=2)

for i in range(10) :
    print(f'\nFold {i}')
    model.fit(X_train,Y_train,epochs=5,callbacks=callbacks)
    model.evaluate(X_dev,Y_dev)