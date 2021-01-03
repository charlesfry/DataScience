# import essentials
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# set the seed for reproducability
seed = 69
random.seed = seed
np.random.seed(seed)

# get data
data = pd.read_csv('../input/high_diamond_ranked_10min.csv')

# data cleaning time!

# Heat map to show which features correlate with y
corrmat = data.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)

# many features are linearly dependent on one another. Gotta drop.

# redGoldDiff is -1 * blueGoldDiff
# blueTotalGold is NOT dropped because it is not a
# linear combination of cs and kills for game-specific reasons
# check if blueFirstBlood and redFirstBlood are opposites
nokills = pd.DataFrame(data['blueFirstBlood'] + data['redFirstBlood'] == 0)
print('\nno kill games:', str(pd.DataFrame.sum(nokills).values[0]) + '\n')
# so if blueFirstBlood is false, redFirstBlood is true and vice-versa
# so drop redFirstBlood

# total experience is not a zero-sum game, so leave redTotalExperience in
# same with dragons, kills, wards placed/destroyed
# but red and blue deaths are almost opposites (not completely opposites since executes can occur
# so get rid of deaths
# also get rid of red total gold and exp, since blue gold - difference = red gold
# blueEliteMonsters = blueDragons + blueHeralds
dropset = ['redGoldDiff', 'redFirstBlood', 'blueDeaths', 'redDeaths',
           'redExperienceDiff', 'redTotalGold', 'redTotalExperience',
           'blueEliteMonsters', 'redEliteMonsters', 'gameId']

outcomes = data.pop('blueWins')
data = data.drop(columns=dropset)

# adding features time!
def square_feats(_columns, _data):
    for col in _columns :
        _data[col + '_2'] = _data[col] ** 2
    return _data

square_cols = [
    'blueWardsPlaced', 'redWardsPlaced', 'blueWardsDestroyed',
    'redWardsDestroyed', 'blueExperienceDiff'
    ]
data = square_feats(square_cols, data)

# check that all the variables are in
print(str(data.dtypes) + '\n')

# observe our dataset
print(data.describe())

# create the train and test sets
from sklearn.model_selection import \
    train_test_split, cross_val_predict, cross_val_score
from sklearn.model_selection import KFold

X_train, X_test, y_train, y_test = train_test_split(
    data, outcomes, test_size=0.20,
        random_state=seed, shuffle=True, stratify=outcomes)

# scale our data
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

# define how we will output scores to the screen:
def score_method(_y_actual :np.array, _y_pred :np.array) :
    errors = np.abs(_y_actual - _y_pred)
    _score = np.sum(errors) / _y_actual.shape[0]
    return 1 - _score

def score_models(_models :dict, _X=X_train, _y=y_train) :
    t_0 = time.time()
    kfold = KFold(n_splits=5, shuffle=False)
    output = list()
    for name, model in _models.items() :
        predictions = cross_val_predict(model, _X, _y, cv=kfold, n_jobs=-1)
        _t_1 = time.time() - t_0
        _pred_score = score_method(y_train,predictions)
        _score = cross_val_score(model, _X, _y, cv=kfold, n_jobs=-1, scoring='roc_auc')
        output.append([name,_pred_score,_t_1])
        print('\n{} ROC Score: {:.4f}, ({:.4f})'
              .format(name, _score.mean(), _score.std()))
        print('{} accuracy: {:.4f}\nTime elapsed: {:.1f}s'
              .format(name,_pred_score,_t_1))
        model.fit(X_train,y_train)
        test_eval = model.score(X_test,y_test)
        print('{} accuracy on the test set: {:.4f}\nVariance: {:.4f}'
              .format(name, test_eval, _score.mean() - _pred_score))
    return output

# regression tiiime
from sklearn.linear_model import ElasticNet, Lasso, \
    BayesianRidge, LassoLarsIC, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# create empty model dictionary that will contain the model under its name
models = {}


# define our models and add them to our model dictionary
logit = make_pipeline(RobustScaler(),
                      LogisticRegression(random_state=seed, max_iter=1000,n_jobs=-1))
models['Logit'] = logit

lasso = make_pipeline(RobustScaler(),
                      Lasso(alpha=.0005, random_state=seed))
models['Lasso'] = lasso

eNet = make_pipeline(RobustScaler(),
                     ElasticNet(alpha=.0005, l1_ratio=.9, random_state=seed))
models['ENet'] = eNet

rforest = RandomForestClassifier(max_depth=5,n_estimators=64)
models['Random Forest'] = rforest

svm = SVC(kernel='rbf',random_state=seed)
models['Basic SVM'] = svm

bayesian_ridge = BayesianRidge()
models['Bayesian Ridge'] = bayesian_ridge

lasso_lars_ic = LassoLarsIC(max_iter=15)
models['LassoLarsIC'] = lasso_lars_ic

score_models(models)

# Let's build our own stacking classifier
from sklearn.base import BaseEstimator, RegressorMixin, \
    TransformerMixin, clone
from sklearn.model_selection import cross_val_predict

print('--------------------------------------')
print('\nBuilding a stacking classifier...\n')

kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

class StackedClassifier(BaseEstimator, RegressorMixin,
                      TransformerMixin):
    def __init__(self, models, final_estimator):
        self.models = models
        self.cloned_models = \
            [clone(x) for x in self.models]
        self.final_estimator = clone(final_estimator)

    # clone original models to fit data into
    def fit(self, X, y):
        # train clones
        predictions = np.column_stack([
            cross_val_predict(model,X,y,cv=kfold,n_jobs=-1,method='predict_proba')
            for model in self.cloned_models
        ])
        for model in self.cloned_models : model.fit(X,y)
        self.final_estimator.fit(predictions, y)
        return self

    # Now execute predictions for cloned models
    def predict(self, X):
        base_predictions = np.column_stack([
            model.predict(X) for
            model in self.cloned_models
        ])
        final_predictions = self.final_estimator.predict(base_predictions)
        return final_predictions

# instantiate our stacking regressor
stacked_classifier = StackedClassifier(
    [model for model in models.values()],
    LogisticRegression(penalty='l2',random_state=seed))

# fit and score our model
stacked_classifier.fit(X_train,y_train)
predictions = stacked_classifier.predict(X_test)

score = score_method(y_test, predictions)
print('\n\nStacked Classifier performance: {}'.format(score))

# now let's try a prepackadged stacking classifier
from sklearn.ensemble import StackingClassifier
from sklearn.svm import LinearSVC

estimators = [
    ('rf', RandomForestClassifier(n_estimators=128,
                                  random_state=seed)),
    ('svr', make_pipeline(LinearSVC(random_state=seed,
                                    max_iter=30000))),
    ('logit', make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000,n_jobs=-1,
                                               random_state=seed))),
    ('Random Forest', RandomForestClassifier(max_depth=5,n_estimators=64)),
    ('SVC',SVC(kernel='rbf',random_state=seed)),
]

print('\nNow evaluating prepackadged Stacking Classifier...')

score = score_method(y_test, predictions)
print('\n\nStacked Classifier performance: {}'.format(score))











quit()
# looks like our Logit model performs best. Let's see if a neural net does any better

#import tensorflow
import tensorflow as tf
from tensorflow.keras import layers

# make tensorflow a little quieter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def score_neural_net(_neural_net :tf.keras.models.Sequential,
                     X_data=X_train, y_data=y_train) :
    n_folds = 5
    kfold = KFold(n_splits=n_folds)
    _score = np.zeros(n_folds)
    for i, (train_index, test_index) in enumerate(kfold.split(X_data)) :
        _X_train,_X_test = X_data[train_index], X_data[test_index]
        _y_train,_y_test = y_data[train_index], y_data[test_index]
        _neural_net.fit(_X_train, _y_train)
        pred = _neural_net.predict(_X_test)
        _score[i] = score_method(_y_test, pred)
    return _score

def build_neural_net() :
    _neural_net = tf.keras.models.Sequential([
        layers.Dense(64,activation='relu'),
        layers.Dense(64,activation='sigmoid'),
        layers.Dense(1,activation='sigmoid')
    ])

    _neural_net.compile(
        optimizer='rmsprop',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'])

    return _neural_net

neural_net = build_neural_net()

print('\nFitting Neural Net...')
neural_net.fit(X_train,y_train,epochs=10)

# add a predictions layer
neural_net.add(
    layers.Lambda(
        lambda x: tf.math.round(x)
    )
)

neural_net.predict(X_test)

print('\nEvaluating model...')
neural_net.evaluate(X_test,y_test)

stacking_clf = StackingClassifier(
    estimators=estimators, final_estimator=
    LogisticRegression(max_iter=10000,random_state=seed),
    n_jobs=-1, verbose=1
)

# evaluate our final model
stacking_clf.fit(X_train,y_train)
score = stacking_clf.score(X_test,y_test)
print('\nStacking Classifier Score on the test set: {:.4f}'
      .format(score))

