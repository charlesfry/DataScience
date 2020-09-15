# import essentials
import os
import random
import pickle
from math import ceil

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE

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
            with open(f'./xgboost/{col}') as hand :
                pipe_dict[col] = pickle.load(hand)
        except FileNotFoundError:
            pass
        try :
            with open(f'./xgboost/rfe/{col}') as hand :
                rfe_dict[col] = pickle.load(hand)
        except FileNotFoundError:
            pass
    try :
        with open('./xgboost/dicts/loss_dict') as hand :
            loss_dict = pickle.load(hand)
    except FileNotFoundError:
        pass

    return pipe_dict,rfe_dict,loss_dict


def append_to_grid(grid,params) :

    for k,v in grid.items() :
        if k in params :
            grid[k] = list(set(v).union({params[k]}))

    return grid


from sklearn.pipeline import Pipeline
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
        row += noise
        new_df = new_df.append(row)

        assert np.isnan(new_df.values).sum() == 0, print(f'NaN values detected: {new_df[new_df.isna().any(axis=1)]}')

    new_df = new_df.sample(frac=1, random_state=seed)

    new_X, new_y = new_df.iloc[:, :-1], new_df.iloc[:, -1]

    assert new_X.shape[0] == new_y.shape[0]
    assert np.isnan(new_X.values).sum() == 0
    assert np.isnan(new_y.values).sum() == 0, f'there are {new_y.isna().sum()} errors'

    return new_X.values, new_y.values

def make_rfe(X,y,rfe_clf,keep_feats=600) :
    step = ceil((X.shape[1] - keep_feats) / 2)
    rfe = RFE(estimator=rfe_clf,n_features_to_select=keep_feats,step=step)
    rfe.fit(X,y)
    return rfe

def make_pipe(clf,rfe,param_grid) :
    pipe = Pipeline([
        ('scaler',StandardScaler()),
        ('rfe',rfe),
        ('clf',clf)
    ])

    grid = GridSearchCV(pipe,param_grid)
    return grid

def fit_pipe(pipe,X_train,y_train) :
    pass


def fit_models(pipe_dict,rfe_dict,loss_dict,train=train,labels=labels,) :

    clean_X = clean_input(train)
    Y = labels

    for col in labels :
        if col == 'sig_id' : continue

        X = clean_X.copy()
        y = Y[col].copy()

# run it

# load dicts
pipe_dict,rfe_dict,loss_dict = load_models()

xgb_params = {'colsample_bytree': 0.6522,
          'gamma': 3.6975,
          'learning_rate': 0.05,
          'max_delta_step': 2.0706,
          'max_depth': 10,
          'min_child_weight': 31.58,
          'n_estimators': 166,
          'subsample': 0.8639
}

xgb = XGBClassifier(**xgb_params)
lgbm = LGBMClassifier(**xgb_params)

param_grid = {
        'colsample_bytree': [69],
        'gamma': [69],
        'learning_rate': [96],
        'max_delta_step': [96],
        'seed':[696969696]
}

def build_dicts(pipe_dict, rfe_dict, loss_dict,train=train,Y=labels,clf=xgb,
                rfe_clf=xgb, param_grid=None) :
    if param_grid is None:
        param_grid = param_grid

    cleaned_input = clean_input(train)

    for col in Y.keys() :
        if col == 'sig_id' : continue

        X = cleaned_input.copy()
        y = Y[col].copy()

        if col in rfe_dict :
            rfe = rfe_dict[col]
        else :
            rfe = make_rfe(X=X,y=y,rfe_clf=rfe_clf,keep_feats=600)

        grid_pipe = make_pipe(clf=clf,)

    return pipe_dict,rfe_dict,loss_dict


pipe_dict,rfe_dict,loss_dict = build_dicts(pipe_dict=pipe_dict,rfe_dict=rfe_dict,loss_dict=loss_dict,train=train,
            Y=labels,clf=xgb,rfe_clf=xgb,param_grid=param_grid)

X = train
print(X[X['cp_type']=='ctl_vehicle'])
print(X[X['cp_type']=='ctl_vehicle'].shape)
print(labels[X['cp_type']=='ctl_vehicle'].sum().sum())