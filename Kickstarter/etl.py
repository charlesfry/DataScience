# extract, transform, and load the data

# first, import our essentials
import numpy as np
import pandas as pd
import os
import pickle
from time import time
import random

def seed_everything(seed=0) :
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    try :
        tf.random.set_seed(seed)
    except :
        pass

seed = 69
seed_everything(seed)

# get our data
def extract(dirs_path) :
    dataset_paths = []
    for root,dirs,files in os.walk(dirs_path) :
        for name in files :
            file = os.path.join(root,name)
            dataset_paths.append(file)
    return dataset_paths


# check that all datasets have the same keys
def check_keys(paths) :
    for i in range(1,len(paths)) :
        df_1 = pd.read_csv(paths[i-1])
        df_2 = pd.read_csv(paths[i])
        if not np.array_equal(df_1.keys(),df_2.keys()) :
            print(i)
            return False
    return True

paths = extract('./input/')

# print(check_keys(paths)) # returns true

df = pd.read_csv(paths[0])

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss,mean_squared_error

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,train_test_split

param_grid = {
    'n_estimators':[100,300],
    'colsample_bytree':[1,.8],
    'max_depth':[6,10]
}

logit = LogisticRegression()
xgb = GridSearchCV(estimator=XGBClassifier(),param_grid=param_grid,scoring='neg_log_loss')

models = {
    'Logistic Regression':logit,
    'XGBoost':xgb
}

def build_pipe(clf) :
    pipe = Pipeline([
        ('scaler',StandardScaler()),
        ('clf',clf)
    ])

    return pipe

from category_encoders import OrdinalEncoder
from pandas.api.types import is_numeric_dtype

def clean_columns(df:pd.DataFrame) :
    numeric_df = df.copy()
    dropped_keys = []
    for col in numeric_df.keys() :
        if not is_numeric_dtype(numeric_df[col]) :
            if numeric_df[col].nunique() > 15 :
                dropped_keys.append(col)
                numeric_df.drop(columns=[col],inplace=True)
            else :
                numeric_df[col] = OrdinalEncoder().fit_transform(numeric_df[col])

    return numeric_df


df.dropna(inplace=True,axis=0)

target = pd.Series((df['pledged'] >= df['goal']),dtype=np.int)
print(target.unique())
df = clean_columns(df)
quit()


print(target.sum()/len(target))

df.drop(columns=['pledged','goal'])


X_train,X_test,y_train,y_test = train_test_split(df,target,random_state=seed)

for name,clf in models.items() :
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_train)
    bias = log_loss(y_test,y_pred)
    y_pred = clf.predict(X_test)
    score = log_loss(y_test, y_pred)

    print(f'{name} bias: {bias}\n'
          f'\tScore: {score}')

