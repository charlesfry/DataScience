import pandas as pd
import numpy as np
import random
import os
import sys
from time import time
import datetime
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_numeric_dtype
import tpot
# is tpot good? gonna find out

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed = 42
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

        # noise = np.append(np.random.randn(1, len(row) - 1), np.array([0])) * .01
        # row += noise
        new_df = new_df.append(row)

    new_df = new_df.sample(frac=1, random_state=seed)

    new_X, new_y = new_df.iloc[:, :-1], new_df.iloc[:, -1]

    return new_X.values, new_y.values

train = pd.read_csv('../input/MoA/train_features.csv')
targets = pd.read_csv('../input/MoA/train_targets_scored.csv')
test = pd.read_csv('../input/MoA/test_features.csv')
submission = pd.read_csv('../input/MoA/sample_submission.csv')

def load_models(labels=targets) :
    tpot_dict = {}
    for col in labels.keys() :
        if col == 'sig_id' : continue
        try:
            with open(f'./input/tpot/{col}', 'r') as hand :
                tpot_dict[col] = tpot
                print(f'Loaded {col}')
        except FileNotFoundError:
            pass
    try :
        with open('./input/tpot/loss_dict', 'rb') as hand :
            loss_dict = pickle.load(hand)
    except FileNotFoundError:
        loss_dict = {}

    return tpot_dict,loss_dict

# drop ctl vehicle
mask = train.iloc[:,1] == 'ctl_vehicle'
train,targets = train.loc[~mask,:],targets.loc[~mask]

from sklearn.model_selection import train_test_split



def build_dicts(tpot_dict, loss_dict, train, targets, reload=False) :
    tpot_mdr_classifier_config_dict = {

        # Classifiers

        'mdr.MDRClassifier': {
            'tie_break': [0, 1],
            'default_label': [0, 1]
        },

        # Feature Selectors

        'skrebate.ReliefF': {
            'n_features_to_select': range(1, 6),
            'n_neighbors': [2, 10, 50, 100, 250, 500]
        },

        'skrebate.SURF': {
            'n_features_to_select': range(1, 6)
        },

        'skrebate.SURFstar': {
            'n_features_to_select': range(1, 6)
        },

        'skrebate.MultiSURF': {
            'n_features_to_select': range(1, 6)
        }

    }

    train = train.copy()
    targets = targets.copy()

    for col in train.keys() :
        if not is_numeric_dtype(train[col]) :
            train[col] = LabelEncoder().fit_transform(train[col])

    for col in targets.keys() :

        if col == 'sig_id' : continue

        if col in tpot_dict :
            print(f'\tAlready fitted {col} with loss {loss_dict[col]}')
            continue

        print(f'Fitting {col}...')

        t = time()

        inp = train.copy().drop(columns=['sig_id'])
        lbls = targets[col].copy()

        if lbls.sum() > 1 :
            inp,lbls = repeat_sample(inp,lbls,2)

        X_train, X_test, y_train, y_test = train_test_split(inp, lbls, stratify=lbls)

        clf = tpot.TPOTClassifier(generations=5, population_size=50, verbosity=3, warm_start=True)

        clf.fit(X_train,y_train)
        loss = clf.score(X_test,y_test)

        tpot_dict[col] = clf
        loss_dict[col] = loss


        clf.export(f'input/tpot/{col}')
        with open(f'input/tpot/loss_dict', 'wb+') as hand:
            pickle.dump(loss_dict, hand)

        print('{}\t\t{}\t\t{:.5f}\n'.format(str(datetime.timedelta(seconds=time() - t))[:7],col, loss))

    total_loss = 0
    for v in loss_dict.values() :
        total_loss += v
    print(f'Average loss: {total_loss / (len(targets.keys())-1)}')

    return tpot_dict,loss_dict


tpot_dict,loss_dict = load_models(labels=targets)

tpot_dict,loss_dict = build_dicts(tpot_dict=tpot_dict, loss_dict=loss_dict, train=train, targets=targets, reload=False)
