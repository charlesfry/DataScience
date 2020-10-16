import tensorflow as tf
tf.keras.backend.set_floatx('float64')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
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

def repeat_sample(X,y,n) :
    assert X.shape[0] == y.shape[0], f'shapes dont match. X shape: {X.shape[1]} y shape: {len(y)}'

    new_X = X.copy()
    new_target = y.copy()

    new_df = pd.DataFrame(np.column_stack((new_X, new_target)))
    repeat_rows = new_df[new_df.iloc[:, -1] == 1]

    start = new_target.sum()

    for i in range(int(start), n):
        row = repeat_rows.sample(frac=1).iloc[0].copy()

        noise = np.append(np.random.randn(1, len(row) - 1), np.array([0])) * .01
        # row += noise
        new_df = new_df.append(row)

    new_df = new_df.sample(frac=1, random_state=seed)

    new_X, new_y = new_df.iloc[:, :-1], new_df.iloc[:, -1]

    return new_X.values, new_y.values

train:pd.DataFrame = pd.read_csv('./input/train_features.csv')
target:pd.DataFrame = pd.read_csv('./input/train_targets_scored.csv')
test:pd.DataFrame = pd.read_csv('./input/test_features.csv')

ctrl_mask = train['cp_type'] == 'ctl_vehicle'

target = target[ctrl_mask]
train = train[ctrl_mask]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization,Dropout,Dense,Input
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from category_encoders import OrdinalEncoder


def build_pipe():
    model = Sequential([

        BatchNormalization(),
        WeightNormalization(Dense(1028, activation='relu', kernel_initializer='he_uniform', )),
        Dropout(.2),

        BatchNormalization(),
        WeightNormalization(Dense(512, activation='relu', kernel_initializer='he_uniform', )),
        Dropout(.2),

        BatchNormalization(),
        WeightNormalization(Dense(206, activation='sigmoid', kernel_initializer='glorot_uniform')),
        Dropout(.2),

        BatchNormalization(),
        WeightNormalization(Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')),
        Dropout(.2),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
    )



    return model


from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import log_loss

def out_of_folds_predict(X, y) :
    callbacks = [
        EarlyStopping(
            # Stop training when loss is no longer improving
            monitor="loss",
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=1e-5,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=2,
            verbose=0,
        )
    ]

    preds = np.zeros(X.shape[0])

    n_splits = 4

    if y.sum() < 2 :
        kfold = KFold(n_splits=n_splits)
    else :
        kfold = StratifiedKFold(n_splits=n_splits)

    for i,(train_index,test_index) in enumerate(kfold.split(X,y)) :
        print(f'Split {i+1} of {n_splits}...')
        pipe = build_pipe()

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        encoder = OrdinalEncoder()
        X_train = encoder.fit_transform(X_train,y_train).astype(np.float)

        pipe.fit(X_train,y_train,epochs=20,callbacks=callbacks,verbose=0)

        X_test = encoder.transform(X_test).astype(np.float)
        pipe.evaluate(X_test,y_test,verbose=1)

        preds[test_index] = pipe.predict(X_test).flatten()

    pipe = build_pipe()

    return preds


models_path = 'E:\DataScience\MoA\models'

def load_models(models_path,target=target) :
    models_dict = {}

    for col in target.keys() :
        if col == 'sig_id': continue
        try:
            models_dict[col] = tf.keras.models.load_model(f'{models_path}/{col}')

        except :
            pass

    try :
        models_dict['final'] = tf.keras.models.load_model(f'{models_path}/final')
    except :
        pass

    try :
        with open(f'{models_path}/scores', 'rb') as handle:
            score_dict = pickle.load(handle)
    except :
        score_dict = {}

    return models_dict,score_dict

callbacks = [
        EarlyStopping(
            # Stop training when loss is no longer improving
            monitor="loss",
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=1e-5,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=2,
            verbose=0,
        )
    ]

def build_preds(models_dict, score_dict, train, target, models_path, seed=0, callbacks=None) :

    if callbacks is None:
        callbacks = callbacks

    try :
        oof_preds = pd.read_csv(f'{models_path}/oof_preds')
    except :
        oof_preds = pd.DataFrame(data=train['sig_id'],columns=['sig_id'])

    for i,col in enumerate(target.keys()) :

        if col == 'sig_id' : continue

        if col in score_dict :
            print(f'\tAlready fitted {col}')
            continue

        print(f'\n{i} of {len(target.keys())-1}\n\tFitting {col}...')
        t = time()

        X = train.iloc[:,1:].copy().values
        y = target[col].values

        pipe = build_pipe()

        #n = 2
        #if y.sum() < n :
        #    X,y = repeat_sample(X=X,y=y,n=n)

        i_preds = out_of_folds_predict(X,y)

        oof_preds[col] = i_preds

        X = OrdinalEncoder().fit_transform(X).astype(np.float)

        pipe = build_pipe()
        pipe.fit(X,y,epochs=20,callbacks=callbacks,verbose=0)

        score = log_loss(target[col],i_preds,labels=[0,1])
        models_dict[col] = pipe
        score_dict[col] = score

        oof_preds.to_csv(f'{models_path}/oof_preds')
        pipe.save(f'{models_path}/{col}')

        with open(f'{models_path}/scores', 'wb') as handle:
            pickle.dump(score_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{models_path}/scores', 'wb+') as handle:
            pickle.dump(score_dict,handle)

        seed += 1
        seed_everything(seed)

        print('\n{}\t\t{}\t{:.6f}\n'.format(str(datetime.timedelta(seconds=time() - t))[:7], col,score_dict[col]))

    iter = 0
    total_score = 0
    for v in score_dict.values() :
        iter += 1
        total_score += v
    print(f'Final Score: {total_score / iter}')

    return score_dict,oof_preds

models_dict,score_dict = load_models(models_path,target)

# score_dict = {}

score_dict,oof_preds = build_preds(models_dict=models_dict,score_dict=score_dict,train=train,
                                               target=target,models_path=models_path,seed=seed)









def final_pipe() :

    model = Sequential([

        BatchNormalization(),
        WeightNormalization(Dense(1028, activation='relu', kernel_initializer='he_uniform', )),
        Dropout(.2),

        BatchNormalization(),
        WeightNormalization(Dense(512, activation='relu', kernel_initializer='he_uniform', )),
        Dropout(.2),

        BatchNormalization(),
        WeightNormalization(Dense(206, activation='sigmoid', kernel_initializer='glorot_uniform')),
        Dropout(.2),

        BatchNormalization(),
        WeightNormalization(Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')),
        Dropout(.2),
    ])

    model.compile(
        optimizer='adam',
        loss='mean_squared_logarithmic_error',
    )

    return model

print(f'OOF predictions score: {log_loss(target,oof_preds,labels=[0,1])}')

# now fit final model
def final_fit(oof, target) :

    preds = np.zeros(oof.shape)

    for tr,te in KFold(n_splits=6).split(oof, target) :
        model = final_pipe()

        X_train,X_test = oof[tr], oof[te]
        Y_train,Y_test = target[tr],target[te]

        model.fit(X_train,Y_train)

        preds[te] = model.predict(X_test)
        print(f'(\t{model.evaluate(X_test, Y_test)})')

    print(f'\nFinal Score: {log_loss(target,preds)}')

final_fit(oof_preds,target)