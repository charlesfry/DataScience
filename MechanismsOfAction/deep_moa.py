# Commented out IPython magic to ensure Python compatibility.
# import essentials
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
print('\n')

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
seed = 69
seed_everything(seed)

train = pd.read_csv('../input/MoA/train_features.csv')
target = pd.read_csv('../input/MoA/train_targets_scored.csv')
test = pd.read_csv('../input/MoA/test_features.csv')

def clean_df(_df) :
    df = _df.copy()
    df = df.drop(columns=['sig_id'])
    df['vehicle'] = df.cp_type.apply(
        lambda x: x == 'ctl_vehicle'
    ).astype(np.int8)
    df.drop(columns=['cp_type'],inplace=True)
    df['d2'] = df.cp_dose.apply(
        lambda x: x == 'D2'
    ).astype(np.int8)
    df.drop(columns=['cp_dose'],inplace=True)
    return df

X = clean_df(train).values
Y = target.iloc[:,1:].values

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, BatchNormalization
from tensorflow_addons.layers import WeightNormalization

from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import log_loss

def build_model() :
    model = Sequential([
        BatchNormalization(),
        WeightNormalization(Dense(2048,activation='relu')),
        Dropout(.4),
        
        BatchNormalization(),
        WeightNormalization(Dense(1028,activation='relu')),
        Dropout(.2),

        BatchNormalization(),
        WeightNormalization(Dense(512,activation='relu')),
        Dropout(.2),

        BatchNormalization(),
        WeightNormalization(Dense(206,activation='sigmoid')),
        Dropout(.2),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return model

model = build_model()

scores = np.empty(0)

n_starts = 3
for seed in range(n_starts) :
    kfold = KFold(n_splits=7,shuffle=True,random_state=seed)

    X_train,X_test,Y_train,Y_test = 0,0,0,0

    for n, (train,test) in enumerate(kfold.split(X)) :
        model = build_model()
        X_train,X_test = X[train],X[test]
        Y_train,Y_test = Y[train],Y[test]

    model.fit(X_train,Y_train,epochs=20)

    bias = model.evaluate(X_train,Y_train)
    score = model.evaluate(X_test,Y_test)
    variance = score - bias

    print(f'Model {n} Fold {seed}:')
    print('Bias: {:.4f}\nVariance: {:.4f}\nScore: {:.4f}'.format(bias,variance,score))
    scores = np.append(scores,np.array([score]))

print(f'\nAveraged Score: {scores.mean()}')
if scores.mean() < .01844 :
    print('YOU POGGERSED IT!!!')
    model = build_model()

    model.fit(X,Y,epochs=20)
    preds = model.predict(test)
    submission = pd.DataFrame(data=np.column_stack((test.sig_id, preds)), columns=target.keys())

    submission.to_csv('../input/MoA/submission.csv', index=False)
    print(pd.read_csv('../input/MoA/submission.csv').head())

