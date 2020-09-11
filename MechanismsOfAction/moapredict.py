# import essentials
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import random
import pickle
from time import time
import datetime

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed = 69
seed_everything(seed)

train = pd.read_csv('../input/MoA/train_features.csv')
labels = pd.read_csv('../input/MoA/train_targets_scored.csv')
test = pd.read_csv('../input/MoA/test_features.csv')
print(f'train shape: {train.shape}')
print(f'target shape: {labels.shape}')

def clean_X(_df) :
    df = _df.drop(columns=['sig_id']).copy()
    df['vehicle'] = df.cp_type.apply(
        lambda x: x == 'ctl_vehicle'
    ).astype(np.int8)
    df.drop(columns=['cp_type'],inplace=True)
    df['d2'] = df.cp_dose.apply(
        lambda x: x == 'D2'
    ).astype(np.int8)
    df.drop(columns=['cp_dose'],inplace=True)
    return df

data = clean_X(train)
target = labels.iloc[:,1:].copy()

# build model

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization,Conv2D,Dropout,Dense
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.models import Sequential
def build_model() :
    """

    :return: built model
    """


    model = Sequential([
        BatchNormalization(),
        Dense()
    ])