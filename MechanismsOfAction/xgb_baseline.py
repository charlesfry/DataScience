import numpy as np
import pandas as pd
import random
import pickle

from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from category_encoders import CountEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss

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
NFOLDS = 5


train = pd.read_csv('../input/MoA/train_features.csv')
targets = pd.read_csv('../input/MoA/train_targets_scored.csv')
test = pd.read_csv('../input/MoA/test_features.csv')
submission = pd.read_csv('../input/MoA/sample_submission.csv')

# drop id col
X = train.iloc[:,1:].to_numpy()
X_test = test.iloc[:,1:].to_numpy()
Y = targets.iloc[:,1:].to_numpy()

# we will make a dictionary to hold all our seperate xgb models
def load_models(labels=targets) :
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

params = {'clf__colsample_bytree': 0.6522,
          'clf__gamma': 3.6975,
          'clf__learning_rate': 0.0503,
          'clf__max_delta_step': 2.0706,
          'clf__max_depth': 10,
          'clf__min_child_weight': 31.5800,
          'clf__n_estimators': 166,
          'clf__subsample': 0.8639
}

