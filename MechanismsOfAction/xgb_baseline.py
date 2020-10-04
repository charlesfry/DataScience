import numpy as np
import pandas as pd
import random
import pickle
import datetime
from time import time

from xgboost import XGBClassifier
from category_encoders import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone

import matplotlib.pyplot as plt

import os
import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed = 69
seed_everything(seed)
nfolds = 5


train = pd.read_csv('../input/MoA/train_features.csv')
targets = pd.read_csv('../input/MoA/train_targets_scored.csv')
test = pd.read_csv('../input/MoA/test_features.csv')
submission = pd.read_csv('../input/MoA/sample_submission.csv')

# drop id col
X_train = train.iloc[:,1:]
X_test = test.iloc[:,1:]
Y = targets.iloc[:,1:]

# we will make a dictionary to hold all our seperate xgb models
def load_models(labels=targets) :
    pipe_dict = {}
    loss_dict = {}

    for col in labels.keys() :
        if col == 'sig_id' : continue
        try:
            with open(f'./input/xgb_baseline/{col}', 'rb') as hand :
                pipe_dict[col] = pickle.load(hand)
                print(f'Loaded {col}')
        except FileNotFoundError:
            pass
    try :
        with open('./input/xgb_baseline/loss_dict', 'rb') as hand :
            loss_dict = pickle.load(hand)
    except FileNotFoundError:
        pass

    return pipe_dict,loss_dict

def append_to_grid(param_grid,params) :
    for k,v in param_grid.items() :
        if k in params :
            param_grid[k] = sorted(list(set(v).union({params[k]})))

    return param_grid

def make_gridsearch(clf,param_grid,params) :
    pipe = Pipeline([
        ('encoder',OrdinalEncoder()),
        ('scaler',StandardScaler()),
        ('clf',clf)
    ])

    pipe.set_params(**params)
    grid = GridSearchCV(estimator=pipe,param_grid=param_grid,cv=3)
    return grid

def make_pipe(clf,params) :
    pipe = Pipeline([
        ('encoder',OrdinalEncoder()),
        ('scaler',StandardScaler()),
        ('clf',clf)
    ])

    pipe.set_params(**params)
    return pipe



def repeat_sample(X,y,n) :
    assert X.shape[0] == y.shape[0], f'shapes dont match. X shape: {X.shape[1]} y shape: {len(y)}'

    new_X = X.copy()
    new_target = y.copy()

    new_df = pd.DataFrame(np.column_stack((new_X, new_target)))
    repeat_rows = new_df[new_df.iloc[:, -1] == 1]

    start = new_target.sum()

    for i in range(int(start), n):
        row = repeat_rows.sample(frac=1).iloc[0].copy()

        # noise = np.append(np.random.randn(1, len(row) - 1), np.array([0])) * .01
        # row += noise
        new_df = new_df.append(row)

    new_df = new_df.sample(frac=1, random_state=seed)

    new_X, new_y = new_df.iloc[:, :-1], new_df.iloc[:, -1]

    return new_X.values, new_y.values

def fit_gridsearch(clf,X,y) :
    """

    :param gridsearch:
    :param X:
    :param y:
    :return: pipe and loss
    """

    n = 2
    if y.sum() < n :
        X,y = repeat_sample(X,y,n)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=seed,stratify=y)

    n = 8
    if y.sum() < n:
        X_tr, y_tr = repeat_sample(X_tr, y_tr, n)

    # drop where cp_type==ctl_vehicle (baseline)
    ctl_mask = X_tr[:, 0] == 'ctl_vehicle'
    X_tr = X_tr[~ctl_mask, :]
    y_tr = y_tr[~ctl_mask]

    ctl_mask = X_te[:, 0] == 'ctl_vehicle'
    X_te = X_te[~ctl_mask,:]
    y_te = np.ravel(y_te[~ctl_mask]).astype(np.int)

    clf.fit(X_tr,y_tr)
    preds = clf.predict_proba(X_te)
    preds = np.ravel(preds[:,1]).astype(np.float)

    loss = log_loss(y_te,preds,labels=[0,1])

    # return clf,loss
    try :
        return clf.best_estimator_,loss
    except :
        return clf,loss

def build_dicts(pipe_dict,loss_dict,targets,params,gridsearch_params,X_train,Y,reload=False) :

    for col in targets.keys() :

        if col == 'sig_id' : continue

        t = time()
        X = X_train.copy().to_numpy()
        y = Y[col].copy()
        param_grid = append_to_grid(param_grid=gridsearch_params, params=params)

        if reload and col in pipe_dict :
            if loss_dict[col] > 0.018 :
                print(f'Refitting {col} with loss {loss_dict[col]}...')
                pipe = clone(pipe_dict[col])
                param_grid = append_to_grid(param_grid=gridsearch_params,params=pipe.get_params())
                clf = make_gridsearch(pipe,param_grid=param_grid,params=pipe.get_params())
                clf, loss = fit_gridsearch(clf, X, y)

                if loss >= loss_dict[col] :
                    print(f'Loss still too high at {loss}')
                    continue

                pipe_dict[col] = clf
                loss_dict[col] = loss

            else :
                loss = loss_dict[col]
                print(f'\t{col} loss good enough at {loss}')
                continue

        elif col in pipe_dict :
            loss = loss_dict[col]
            print(f'\t{col} already fitted at loss {loss}')
            continue


        else :
            print(f'Fitting {col}...')
            xgb = XGBClassifier()


            clf = make_pipe(clf=xgb, params=params)
            #clf = make_gridsearch(clf=xgb,param_grid=param_grid,params=params)

            clf,loss = fit_gridsearch(clf,X,y)

            print(type(clf))
            pipe_dict[col] = clf
            loss_dict[col] = loss

        with open(f'input/xgb_baseline/{col}', 'wb+') as hand:
            pickle.dump(pipe_dict[col], hand)
        with open(f'input/xgb_baseline/loss_dict', 'wb+') as hand:
            pickle.dump(loss_dict, hand)

        print(f'\t{col} final params:\n{pipe_dict[col].named_steps["clf"].get_params()}\n')

        print('{}\t\t{}\t\t{:.5f}\n'.format(str(datetime.timedelta(seconds=time() - t))[:7],col, loss))

    total_loss = 0
    for v in loss_dict.values() :
        total_loss += v
    print(f'Average loss: {total_loss / (len(targets.keys())-1)}')

    return pipe_dict,loss_dict


params = {'clf__colsample_bytree': 0.6522,
          'clf__gamma': 3.6975,
          'clf__learning_rate': 0.0503,
          'clf__max_delta_step': 2.0706,
          'clf__max_depth': 10,
          'clf__min_child_weight': 31.5800,
          'clf__n_estimators': 166,
          'clf__subsample': 0.8639,
          #'clf__lambda': 1
}

gridsearch_params = {
    'clf__lambda' : [1,3,9],
    'clf__n_estimators':[300]
}

pipe_dict,loss_dict = load_models(labels=targets)

pipe_dict,loss_dict = build_dicts(pipe_dict=pipe_dict,loss_dict=loss_dict,targets=targets,params=params,
                                             gridsearch_params=gridsearch_params,X_train=X_train,Y=Y,reload=True)
