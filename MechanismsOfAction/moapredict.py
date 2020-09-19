# import essentials
import os
import random
import pickle
from math import ceil
import datetime
from time import time

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.metrics import log_loss

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    #try :
    #    tf.random.set_seed(seed)
    #except :
    #    pass

seed = 69
seed_everything(seed)

train = pd.read_csv('../input/MoA/train_features.csv')
labels = pd.read_csv('../input/MoA/train_targets_scored.csv')
test = pd.read_csv('../input/MoA/test_features.csv')
print(f'train shape: {train.shape}')
print(f'target shape: {labels.shape}')

def clean_input(_df) :
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



def load_models(labels=labels) :
    pipe_dict = {}
    rfe_dict = {}
    loss_dict = {}

    for col in labels.keys() :
        try:
            with open(f'input/{col}', 'rb') as hand :
                pipe_dict[col] = pickle.load(hand)
                print(f'Loaded {col}')
        except FileNotFoundError:
            pass
        try :
            with open(f'input/rfe/{col}', 'rb') as hand :
                rfe_dict[col] = pickle.load(hand)
        except FileNotFoundError:
            pass
    try :
        with open('input/dicts/loss_dict', 'rb') as hand :
            loss_dict = pickle.load(hand)
    except FileNotFoundError:
        pass
    print('Loaded dictionaries...\n\n\n')
    return pipe_dict,rfe_dict,loss_dict


def append_to_grid(grid,params) :

    for k,v in grid.items() :
        if k[5:] in params : # clf__estimator__ = 16 characters
            grid[k] = list(set(v).union({params[k[16:]]}))

    return grid

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

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

    assert np.isnan(new_df.values).sum() == 0, print(f'NaN values detected: {new_df[new_df.isna().any(axis=1)]}')

    new_df = new_df.sample(frac=1, random_state=seed)

    new_X, new_y = new_df.iloc[:, :-1], new_df.iloc[:, -1]

    assert new_X.shape[0] == new_y.shape[0]
    assert np.isnan(new_X.values).sum() == 0
    assert np.isnan(new_y.values).sum() == 0, f'there are {new_y.isna().sum()} errors'

    return new_X.values, new_y.values

def make_rfe(X_train,y_train,rfe_clf,keep_feats=600) :
    print(f'\tFitting RFE')
    step = ceil((X_train.shape[1] - keep_feats) / 2)
    rfe = RFE(estimator=rfe_clf,n_features_to_select=keep_feats,step=step)
    rfe.fit(X_train,y_train)
    print('\tRFE fit successfully')
    return rfe

def make_pipe(clf,param_grid) :
    pipe = Pipeline([
        ('scaler',StandardScaler()),
        ('selectkbest', SelectKBest(score_func=f_classif,k=100)),
        ('smote',SMOTE(random_state=seed,k_neighbors=5)),
        ('clf',clf)
    ])

    grid = GridSearchCV(pipe,param_grid,cv=3,error_score='raise')
    return grid

from sklearn.model_selection import train_test_split

def split_X_y(X,y) :
    n = 2
    if y.sum() < n:
        X, y = repeat_sample(X, y, n)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, stratify=y)

    n = 28
    if y.sum() < n:
        print(f'\tOnly {y_train.sum()} samples in y_train. Increasing to {n}...')
        X_train, y_train = repeat_sample(X_train, y_train, n)

    return X_train,X_test,y_train,y_test

def fit_score_pipe(pipe,X_train,X_test,y_train,y_test) :

    assert np.isnan(X_train).sum().sum() == 0, f'{np.isnan(X_train).sum()}'
    assert np.isnan(X_test).sum().sum() == 0
    assert np.isnan(y_train).sum().sum() == 0
    assert np.isnan(y_test).sum().sum() == 0

    pipe.fit(X_train,y_train)
    pred = pipe.predict(X_train)
    bias = log_loss(y_train,pred,labels=[0,1])
    pred = pipe.predict(X_test)
    loss = log_loss(y_test,pred,labels=[0,1])
    best_params = pipe.best_estimator_.get_params()

    for k in pipe.param_grid :
        print(f'{k}:{best_params[k]}')

    return pipe.best_estimator_,loss,bias

# run it

# load dicts
pipe_dict,rfe_dict,loss_dict = load_models()

xgb_params = {
              # 'colsample_bytree': 0.6522,
              # 'gamma': 3.6975,
              # 'learning_rate': 0.05,
              # 'max_delta_step': 2.0706,
              # 'max_depth': 10,
              # 'min_child_weight': 31.58,
              'n_estimators': 166,
              'seed':seed,
              # 'objective':'binary:logistic',
}

lgbm_params = {
    'num_leaves': 300,
    #'min_child_weight': 0.03,
    #'objective': 'binary',
    'max_depth': 8,
    'learning_rate': 0.005,
    "boosting_type": "gbdt",
    "bagging_seed": seed,
    #"metric": 'binary_logloss',
    "verbosity": 0,
    'random_state': seed
}

xgb = XGBClassifier(**xgb_params)
lgbm = LGBMClassifier(**lgbm_params)
rfe_clf = XGBClassifier(**xgb_params)

param_grid = {
        'clf__colsample_bytree': [.5,.8],
        #'clf__gamma': [3,4],
        'clf__max_depth':[5,8],
        'clf__reg_lambda':[3,7,9]
}

def build_dicts(pipe_dict, rfe_dict, loss_dict,train=train,Y=labels,clf=xgb,
                rfe_clf=rfe_clf, param_grid=None, reload=None) :

    if param_grid is None:
        param_grid = param_grid

    cleaned_input = clean_input(train)

    for col in Y.keys() :
        if col == 'sig_id' : continue
        t = time()
        print(f'Evaluating {col}...')

        X = cleaned_input.copy()
        y = Y[col].copy()

        X_train,X_test,y_train,y_test = split_X_y(X,y)

        # if col in pipe_dict and reload is None :
        #     rfe = rfe_dict[col]
        # else :
        #     rfe = make_rfe(X_train,y_train,rfe_clf=rfe_clf,keep_feats=600)
        #     rfe_dict[col] = rfe

        # X_train,X_test = rfe.transform(X_train),rfe.transform(X_test)

        if col in pipe_dict and reload is None :
            print(f'\tAlready fitted {col}')
            pipe = pipe_dict[col]
            loss = loss_dict[col]
            bias = None
        else :
            grid_pipe = make_pipe(clf=clf,param_grid=param_grid)
            pipe, loss, bias = fit_score_pipe(grid_pipe,X_train,X_test,y_train,y_test)

        assert loss >= 0, f'\nError! {col} loss is {loss}'

        pipe_dict[col] = pipe
        loss_dict[col] = loss

        with open(f'input/{col}', 'wb+') as hand:
            pickle.dump(pipe_dict[col], hand)
        # with open(f'./xgboost/rfe/{col}', 'wb+') as hand:
        #     pickle.dump(rfe, hand)
        with open(f'input/dicts/loss_dict', 'wb+') as hand:
            pickle.dump(loss_dict, hand)

        if bias : print(f'\t{col} bias: {bias}')
        print('{}\t\t{}\t\t{:.5f}\n'
              .format(str(datetime.timedelta(seconds=time() - t))[:7],
                      col, loss))

    print(f'\nFinal Score: {np.mean(list(loss_dict.values()))}')

    return pipe_dict,rfe_dict,loss_dict

xgb = XGBClassifier(**xgb_params)

clf = XGBClassifier(**xgb_params)

pipe_dict,rfe_dict,loss_dict = build_dicts(pipe_dict=pipe_dict,rfe_dict=rfe_dict,loss_dict=loss_dict,train=train,
            Y=labels,clf=clf,rfe_clf=rfe_clf,param_grid=param_grid)

