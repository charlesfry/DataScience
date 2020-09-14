# Commented out IPython magic to ensure Python compatibility.
# import essentials
import os
import random
import pickle
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

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

def evaluate(model) :
    kfold = MultilabelStratifiedShuffleSplit(n_splits=4, random_state=seed,test_size=.2)
    scores = np.empty(0)
    for train,test in kfold.split(X,Y) :
        X_train,X_test = X[train],X[test]
        Y_train,Y_test = Y[train],Y[test]
        model.fit(X_train,Y_train,epochs=14)
        scores = np.append(scores,model.evaluate(X_test,Y_test))

    return scores.mean()

model = build_model()
score = evaluate(model)
print(f'Score: {score}')

model = build_model()

model.fit(X,Y,epochs=14)
preds = model.predict(clean_df(test))
submission = pd.DataFrame(data=np.column_stack((test.sig_id, preds)), columns=target.keys())

#submission.to_csv('../input/MoA/submission.csv', index=False)
#print(pd.read_csv('../input/MoA/submission.csv').head())

model.save('./saved_models/deepmoa/deepmoa')