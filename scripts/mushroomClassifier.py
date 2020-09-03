# classify mushrooms as either poisonous or non-poisonous
# using data from https://www.kaggle.com/uciml/mushroom-classification

# import essentials
import numpy as np
import pandas as pd

# set randomness for reproducability
seed = 42
np.random.seed(seed)

# get our data
data = pd.read_csv("../input/mushrooms.csv")

# give it a quick overview
print(data.describe())

# use the documentation to clean our data

# check data types of our data
print('\ndata types of our dataset')
print(str(data.dtypes) + '\n')

# unfortunately our variables are all classified as
# objects with string names, but we need most of them as categories

# grab our column names to iterate over
columns = data.keys()
# changing almost all to categorical variables
# will make a few exceptions
column_exceptions = ['class','bruises']

# convert our target variable "class" to False for
# non-poisonous and True for poisonous
# do the same for bruises
newData = pd.DataFrame()
newData['class'] = (data['class'] == 'p')
newData['class'] = newData['class'].astype(int)
newData['bruises'] = data['bruises'] == 't'

# change all non-exception columns to categorical vars
data.drop(columns=['class','bruises'])
dummy_data = pd.get_dummies(data,drop_first=True, dtype='int')
newData = pd.concat([newData['class'],newData['bruises'],dummy_data],axis=1)
data = newData

# check dtypes again
print('\nCleaned data types')
print(data.dtypes)
# Success!

# next we split into train / test data for evaluation
from sklearn.model_selection import train_test_split

data_independent = data.iloc[:, 1:]  # X
data_dependent = data.iloc[:, 0] # y
X_train, X_test, y_train, y_test = train_test_split(
    data_independent,data_dependent.values,shuffle=True,
    random_state=seed, test_size=.2, stratify=data_dependent)

# import scoring metrics for evaluation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# define our custom scoring method
def log_loss(model, X=X_train, y=y_train, _scoring='neg_log_loss') :
    _kfold = KFold(n_splits=5)
    _score = -cross_val_score(model, X, y, cv=_kfold, scoring=_scoring)
    return _score

def final_accuracy(model,_X_train=X_train,_y_train=y_train,
                   _X_test=X_test,_y_test=y_test) :
    model.fit(_X_train,_y_train)
    _y_hat = model.predict(_X_test)
    _final_score = np.sum(_y_hat == _y_test) / len(_y_test)
    return _final_score

# now make linear estimations and evaluate if we need
# a more complex model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# define models
print('\nLoading and scoring models...')
logit = LogisticRegression(random_state=seed)
svc = SVC(probability=True,random_state=seed,kernel='rbf')
forest = RandomForestClassifier(random_state=seed)

# using log loss, what score do we get on our test set?
score = log_loss(logit)
print('\nLogit score: {:.4f} ({:.4f})'
      .format(score.mean(),score.std()))
score = log_loss(svc)
print('\nSVC score: {:.4f} ({:.4f})'
      .format(score.mean(),score.std()))
score = log_loss(forest)
print('\nRandom Forest score: {:.4f} ({:.4f})'
      .format(score.mean(),score.std()))

# extremely accurate scores on all. What's our prediction rate?
accuracy = final_accuracy(logit)
print('\nLogit final accuracy: {:.4f}'.format(accuracy))
accuracy = final_accuracy(svc)
print('\nSVC final accuracy: {:.4f}'.format(accuracy))
accuracy = final_accuracy(forest)
print('\nRandom Forest final accuracy: {:.4f}'.format(accuracy))

# 100% accuracy on several methods! Wow, machines are great mycologists!
# let's analyze the data to get some intuitions on our results
# let's use our logostic regression since it got everything
# right while still being quite fast
logit.fit(X_train,y_train)
predictions = logit.predict(X_test)

# now perform PCA to view how our machine
# seperates the data
from sklearn.decomposition import PCA

# create our PCA and fit it to the data
pca = PCA(n_components=2, random_state=seed)
pca.fit(X_test)
X_test_pca = pca.transform(X_test)

# create our PCA dataframes
pca_df = pd.DataFrame(data=X_test_pca, columns=['PCA 1', 'PCA 2'])
y_test_series = pd.DataFrame(y_test, columns=['target'])
final_df = pd.concat([pca_df, y_test_series],axis=1)

# import our plotting methods
import matplotlib.pyplot as plt

# create our figure
fig :plt.Figure = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_title('PCA Graph', fontsize=20)

targets = [0, 1]
colors = ['b','r']

for target, color in zip(targets, colors) :
    kept_indicies = predictions == target
    ax.scatter(final_df.loc[kept_indicies, 'PCA 1'],
               final_df.loc[kept_indicies, 'PCA 2'],
               c = color,
               s= 50)
ax.legend(['Non-Poisonous','Poisonous'])
ax.grid()
plt.show()
