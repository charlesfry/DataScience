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

train = pd.read_csv('./input/train_features.csv')
target = pd.read_csv('./input/train_targets_scored.csv')
test = pd.read_csv('./input/test_features.csv')

target = target[train['cp_type']!='ctl_vehicle']
train = train[train['cp_type'] != 'ctl_vehicle']

print(train.head())
def clean_df(_df) :
    df = _df.copy()
    df.set_index('sig_id')
    df['cp_type'] = df['cp_type'].apply(
        lambda x: x == 'trt_cp'
    ).astype(np.int8)
    df['cp_dose'] = df.cp_dose.apply(
        lambda x: x == 'D2'
    ).astype(np.int8)
    return df

X = clean_df(train).values
print(X)
quit()
Y = target.iloc[:,1:].values

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, BatchNormalization
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.regularizers import L1

from sklearn.metrics import log_loss

def build_model() :


    model = Sequential([
        
        BatchNormalization(),
        WeightNormalization(Dense(1028,activation='relu',kernel_initializer='he_uniform',)),
        Dropout(.2),

        BatchNormalization(),
        WeightNormalization(Dense(512,activation='relu',kernel_initializer='he_uniform',)),
        Dropout(.2),

        BatchNormalization(),
        WeightNormalization(Dense(206,activation='sigmoid',kernel_initializer='glorot_uniform')),
        Dropout(.2),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
    )

    return model

def evaluate(model) :
    callbacks = [
        EarlyStopping(
            # Stop training when loss is no longer improving
            monitor="loss",
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=1e-5,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=2,
            verbose=1,
        )
    ]

    kfold = MultilabelStratifiedShuffleSplit(n_splits=4, random_state=seed,test_size=.2)
    scores = np.empty(0)
    for train,test in kfold.split(X,Y) :
        X_train,X_test = X[train],X[test]
        Y_train,Y_test = Y[train],Y[test]

        model.fit(
            X_train,
            Y_train,
            epochs=1,
            #epochs=50,
            callbacks=callbacks,
        ) # change epochs to 14

        Y_pred = model.predict(X_test)
        control_mask = X_test['cp_type'] == 'ctl_vehicle'
        Y_pred[control_mask,:] = 0
        # scores = np.append(scores,model.evaluate(X_test,Y_test))
        scores = np.append(scores,log_loss(Y_test,Y_pred,labels=[0,1]))
    return scores.mean()

model = build_model()
score = evaluate(model)
print(f'Score: {score}')

preds = model.predict(clean_df(test))

submission = pd.DataFrame(data=np.column_stack((test.sig_id, preds)), columns=target.keys())
control_mask = test['cp_type'] == 'ctl_vehicle'
submission[control_mask] = 0
#submission.to_csv('./input/MoA/submission.csv', index=False)
#print(pd.read_csv('./input/MoA/submission.csv').head())

#model.save('./saved_models/deepmoa/deepmoa')