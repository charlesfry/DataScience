# import essentials
import numpy as np
import pandas as pd
import pickle
import random
import os
from time import time
import datetime

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed = 69
seed_everything(seed)

train = pd.read_csv('../input/MoA/train_features.csv')
labels = pd.read_csv('../input/MoA/train_targets_scored.csv')
test = pd.read_csv('../input/MoA/test_features.csv')

def load_models(labels=labels) :
    pipe_dict = {}
    loss_dict = {}

    for col in labels.keys() :
        if col == 'sig_id' : continue
        try:
            with open(f'./input/dicts/{col}', 'rb') as hand :
                pipe_dict[col] = pickle.load(hand)
                print(f'Loaded {col}')
        except FileNotFoundError:
            pass
    try :
        with open('./input/dicts/loss_dict', 'rb') as hand :
            loss_dict = pickle.load(hand)
    except FileNotFoundError:
        pass

    return pipe_dict,loss_dict


def append_to_grid(param_grid,params) :
    for k,v in param_grid.items() :
        if k in params : # clf__estimator__ = 16 characters
            param_grid[k] = sorted(list(set(v).union({params[k]})))

    return param_grid

params = {
    'clf__colsample_bytree': 0.6522,
    'clf__gamma': 3.6975,
    'clf__learning_rate': 0.05,
    'clf__max_delta_step': 2.0706,
    'clf__max_depth': 10,
    'clf__min_child_weight': 31.5800,
    'clf__n_estimators': 166,
    'clf__subsample': 0.8639
}

param_grid = {
    'clf__colsample_bytree': [0.4], # tried [.4]
    'clf__gamma': [5,9], # tried [5,9]
    # 'clf__estimator__max_delta_step': [2.0706],
    'clf__max_depth': [14], # tried [14]
    # 'clf__estimator__min_child_weight': [31.5800],
    # 'clf__n_estimators': [300], # tried [300]
    'clf__subsample': [0.6] # tried [.6]
}

from sklearn.pipeline import Pipeline
from category_encoders import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def make_gridsearch(param_grid) :
    clf_params = {
        'colsample_bytree': 0.6522,
        'gamma': 3.6975,
        'learning_rate': 0.05,
        'max_delta_step': 2.0706,
        'max_depth': 10,
        'min_child_weight': 31.5800,
        'n_estimators': 166,
        'subsample': 0.8639
    }

    pipe = Pipeline([
        ('encoder',OrdinalEncoder()),
        ('scaler',StandardScaler()),
        ('clf',XGBClassifier(**clf_params))
    ])

    gridsearch = GridSearchCV(pipe,param_grid=param_grid,scoring='neg_log_loss',cv=3)

    return gridsearch

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

def fit_gridsearch(gridsearch,X,y) :
    """

    :param grid:
    :param X:
    :param y:
    :return: pipe and loss
    """
    if y.sum() > 1 :
        X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=seed,stratify=y)
    else :
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    gridsearch.fit(X_train,y_train)
    preds = gridsearch.predict(X_test)
    return gridsearch.best_estimator_,log_loss(y_test,preds)




def build_dicts(pipe_dict,loss_dict,train,labels,param_grid,params) :

    X = train.iloc[:,1:].copy()
    X = X[train['cp_type']!='ctl_vehicle']
    Y = labels[train['cp_type']!='ctl_vehicle']


    param_grid = append_to_grid(param_grid=param_grid,params=params)

    for col in labels.keys() :
        if col == 'sig_id' : continue

        t = time()

        if col in pipe_dict :
            print(f'\t{col} already scored at {loss_dict[col]}')
            continue

        print(f'Fitting {col}...')

        y = Y[col].copy()

        gridsearch = make_gridsearch(param_grid=param_grid)

        pipe,loss = fit_gridsearch(gridsearch=gridsearch,X=X,y=y)

        assert loss >= 0, f'\nError! {col} loss is {loss}'

        print(f'\t{pipe.get_params()}')

        print(f'\tfitted {col} with score {loss}')

        pipe_dict[col] = pipe
        loss_dict[col] = loss

        with open(f'input/dicts/{col}', 'wb+') as hand:
            pickle.dump(pipe_dict[col], hand)
        with open(f'input/dicts/loss_dict', 'wb+') as hand:
            pickle.dump(loss_dict, hand)

        print('{}\t\t{}\t\t{:.5f}\n'.format(str(datetime.timedelta(seconds=time() - t))[:7],col, loss))

    total_loss = 0
    cols = 0
    for v in loss_dict.values() :
        total_loss += v
        cols += 1

    print(f'\nAverage score: {total_loss / cols}')

    return pipe_dict,loss_dict

def baseline_run() :
    clf_params = {
        'colsample_bytree': 0.6522,
        'gamma': 3.6975,
        'learning_rate': 0.05,
        'max_delta_step': 2.0706,
        'max_depth': 10,
        'min_child_weight': 31.5800,
        'n_estimators': 166,
        'subsample': 0.8639
    }
    for col in labels.keys() :

        if col == 'sig_id' : continue
        X_train,X_test,y_train,y_test = train_test_split(train.iloc[:,1:],labels[col])
        pipe = Pipeline([
            ('encoder',OrdinalEncoder()),
            ('scaler',StandardScaler()),
            ('clf',XGBClassifier(**clf_params))
        ])

        pipe.fit(X_train,y_train)
        y_pred = pipe.predict(X_test)

        print(f'{col}: {log_loss(y_test, y_pred)}')


pipe_dict,loss_dict = load_models(labels)

pipe_dict,loss_dict = build_dicts(pipe_dict=pipe_dict,loss_dict=loss_dict,train=train,labels=labels,
                                  param_grid=param_grid,params=params)