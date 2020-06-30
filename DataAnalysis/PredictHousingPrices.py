# Thanks to Serigne at Kaggle for advising this code
# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

import sys
print("Python version")
print (sys.version)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style("darkgrid")
from scipy import stats
from scipy.stats import norm, skew
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
# Limited floats output to 3 decimal points

pd.set_option('display.float_format', lambda x: '{:.3f}'\
              .format(x)) #Limiting floats output to 3 decimal points

# ignore annoying warning (from sklearn and seaborn)
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p
# ----------------------------------------------------------------------------------------------------------------------

# get our data, the first is labeled, the second is unlabeled, and we
#   are trying to predict the second
train = pd.read_csv(
    "../input/Kaggle/HousingPrices/train.csv")
test = pd.read_csv(
    "../input/Kaggle/HousingPrices/test.csv")

# Save the ID column
train_ID = train['Id']
test_ID = test['Id']

# Drop the ID column
train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)

# Check to see if there are issues
print("\nThe train data size after dropping Id feature is : {} "
      .format(train.shape))
print("The test data size after dropping Id feature is : {} "
      .format(test.shape))

# The documentation for this dataset indicates outliers
# Checking for outliers here
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel("SalePrice",fontsize=13)
plt.xlabel("GrLivArea",fontsize=13)
plt.title("Original Data")
plt.show()

# This shows two massive outliers in the bottom right corner
# We will delete them to improve accuracy
train = train.drop(train[(train['GrLivArea']>4000) & \
                         (train['SalePrice']<300000)].index)

# Check again to verify that data is removed
_, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel("SalePrice",fontsize=13)
plt.xlabel("GrLivArea",fontsize=13)
plt.title("After data removal")
plt.show()

# our y var is 'SalePrice'. Let's analyze the data first:
sns.distplot(train['SalePrice'], fit=norm)
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# Plot the distribution
plt.legend(['Normal dist. ($\mu$ {:.2f} and $\sigma=$ {:.2f} )'\
           .format(mu,sigma)], loc='best')
plt.ylabel("Frequency")
plt.title('SalePrice distribution')

# Get the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'],plot=plt)
plt.show()

# The distribution is skewed right and not a great fit
# Take the log to fit it better
train['SalePrice'] = np.log1p(train['SalePrice'])

# Assess our new distribution
sns.distplot(train['SalePrice'],fit=norm)

# Get our new fitted parameters
(mu, sigma) = norm.fit(train['SalePrice'])
print('\nNormal dist. (mu = {:.2f} and sigma = {:.2f} )'\
      .format(mu,sigma))

# Plot the dist
plt.legend(['Normal dist. ($\mu$ {:.2f} and $\sigma=$ {:.2f} )'\
            .format(mu,sigma)],
           loc='best')
plt.ylabel("Frequency")
plt.title("SalePrice Distribution")

# Get another QQ-plot
fig = plt.figure()
res = stats.probplot(train["SalePrice"], plot=plt)
plt.show()

# Features Engineering
# Start by concatenating train and test data
m_test = test.shape[0]
y_train = train.SalePrice.values
m_train = train.shape[0]
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is {}".format(all_data.shape))

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na\
    [all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Percent' :all_data_na})
print(missing_data.head(20),'\n\n\n')

f, ax = plt.subplots(figsize=(15,12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features',fontsize=15)
plt.ylabel('Percent missing data',fontsize=15)
plt.title('Percent missing data per feature',fontsize=15)
plt.show()

# Heat map to show which features correlate with y
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()

# Impute missing values
# PoolQC: description says 'na' means no pool
# So set that to None
all_data['PoolQC'] = all_data['PoolQC'].fillna("None")

# MiscFeature, Alley, Fence, FireplaceQu say the same
all_data['MiscFeature'] = \
    all_data['MiscFeature'].fillna("None")
all_data['Alley'] = \
    all_data['Alley'].fillna("None")
all_data['Fence'] = \
    all_data['Fence'].fillna("None")
all_data['FireplaceQu'] = \
    all_data['FireplaceQu'].fillna("None")

# Also true of GarageType, GarageFinish, GarageQual,
# GarageCond
all_data['GarageType'] = \
    all_data['GarageType'].fillna("None")
all_data['GarageFinish'] = \
    all_data['GarageFinish'].fillna("None")
all_data['GarageQual'] = \
    all_data['GarageQual'].fillna("None")
all_data['GarageCond'] = \
    all_data['GarageCond'].fillna("None")

# LotFrontage is trickier. Missing data is just uncollected.
# So guesstimate using similar houses
all_data["LotFrontage"] = all_data.groupby("Neighborhood")\
    ['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# Replace GarageYrBlt, GarageArea and GarageCars missing
# values with 0 since that corresponds with none
for col in ('GarageYrBlt','GarageArea','GarageCars') :
    all_data[col] = all_data[col].fillna(0)

# Same goes for BsmtFinSF1, BsmtFinSF2, BsmtUnfSF,
# TotalBsmtSF, BsmtFullBath and BsmtHalfBath
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
            'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)

# Also for BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1
# and BsmtFinType2
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure',
            'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

# MasVnrArea and MasVnrType also likely mean no veneer,
# so fill them as None and 0 respectively
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)

# Check MSZoning for trends
sns.set(style='darkgrid')
ax = sns.countplot(x=all_data['MSZoning'], data=all_data)
plt.show()

# MSZoning has 'RL' as the most common feature by far
# So just fill it in as RL for the missing data
all_data['MSZoning'] = all_data['MSZoning'].fillna(
    all_data['MSZoning'].mode()[0])

# Utilities is almost all 'AllPub' with 3 exceptions
# So this data seems unhelpful
all_data = all_data.drop(['Utilities'], axis=1)

# Functional description says NA means typical ('Typ')
all_data['Functional'] = all_data['Functional'].fillna('Typ')

# Electrical, KitchenQual, Exterior1st and Exterior2nd
# only have 1 missing variable
# Filling in the mode of each
all_data['Electrical'] = all_data['Electrical'].fillna(
    all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(
    all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(
    all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(
    all_data['Exterior2nd'].mode()[0])

# Fill SaleType with mode again
all_data['SaleType'] = all_data['SaleType'].fillna(
    all_data['SaleType'].mode()[0])

# MSSubClass NA most likely means no building class
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

# Check if any missing data remains
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(
    all_data_na[all_data_na == 0].index)\
        .sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

if len(all_data_na.index) > 0 :
    print('Missing data after cleaning:\n',missing_data.head())
else : print('Data was successfully cleaned with no \
missing entries')

# More Features Engineering
# Some variables are numerical when they should be categorical
# i.e. MSSubClass, OverallCond, YrSold, MoSold
columns = ['MSSubClass','OverallCond', 'YrSold', 'MoSold']
for col in columns :
    all_data[col] = all_data[col].astype(str)

# Label Encode our categorical variabels
columns = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
           'ExterQual', 'ExterCond','HeatingQC', 'PoolQC',
           'KitchenQual', 'BsmtFinType1','BsmtFinType2',
           'Functional', 'Fence', 'BsmtExposure',
           'GarageFinish', 'LandSlope','LotShape',
           'PavedDrive', 'Street', 'Alley', 'CentralAir',
           'MSSubClass', 'OverallCond','YrSold', 'MoSold')
for col in columns :
    lbl = LabelEncoder()
    lbl.fit(list(all_data[col].values))
    all_data[col] = lbl.transform(list(all_data[col].values))

# verify the shape
print(f'Shape of all_data: {all_data.shape}') # (2917,78)

# Adding our final important features
# First is Total Square Footage for its obvious importance
all_data['TotalSF'] = all_data['TotalBsmtSF'] + \
    all_data['2ndFlrSF']

# Skew Corrections features
# Check for skew
numeric_feats = all_data.dtypes[\
    all_data.dtypes != "object"].index # All numeric columns
skewed_feats = all_data[numeric_feats].apply(
    lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew':skewed_feats})
print(skewness.head(10))

# Use Box-Cox Transformation
skewness = skewness[abs(skewness) > 0.75]
print("Number of skewed numerical features: ",
      skewness.shape[0])
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features :
    all_data[feat] = boxcox1p(all_data[feat], lam)
# All skewed data has been Box-Cox transformed

# Now convert categorical variables to dummy variables
print('Data shape before categorical variables converted to \
dummies:', all_data.shape)
all_data = pd.get_dummies(all_data)
print('Data shape with dummies:', all_data.shape)
print(all_data.shape) # 2917 x 220

# Now that we've cleaned the data, it's time to put it back
# into the training and test sets
train = all_data[:m_train]
test = all_data[m_train:]

# Importing the modelling software
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, \
    RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

n_folds = 5

# LEEEROOOOOOOYYYYYYYYYY JENKINNNNSSSSSSSSS

# Shuffle the dataset
def rmsle_cv(model) :
    kf = KFold(n_folds, shuffle=True, random_state=42)\
        .get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values,
        y_train, scoring="neg_mean_squared_error", cv=kf))
    return rmse

# Run some regressions
lasso = make_pipeline(RobustScaler(),
                      Lasso(alpha=.0005, random_state=1))
ENet = make_pipeline(RobustScaler(),
        ElasticNet(alpha=.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=.6, kernel='polynomial',
                    degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000,
    learning_rate=.05, max_depth=4, max_features='sqrt',
    min_samples_leaf=15, min_samples_split=10, loss='huber',
    random_state=5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

score = rmsle_cv(lasso)
print('\nLasso Score: {:.4f} ({:.4f})'
      .format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print('\nElasticNet Score: {:.4f} ({:.4f})'
      .format(score.mean(), score.std()))

score = rmsle_cv(KRR)
print("\nKernel Ridge score: {:.4f} ({:.4f})\n"
      .format(score.mean(), score.std()))

score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n"
      .format(score.mean(), score.std()))

score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n"
      .format(score.mean(), score.std()))

score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n"
      .format(score.mean(), score.std()))


class AveragingModels(BaseEstimator, RegressorMixin,
                      TransformerMixin) :
    def __init__(self, models):
        self.models = models
        self.cloned_models = \
            [clone(x) for x in self.models]

    # clone original models to fit data into
    def fit(self, X, y):
        # train clones
        for model in self.cloned_models:
            model.fit(X,y)

        return self

    # Now execute predictions for cloned models
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for
            model in self.cloned_models
        ])
        return np.mean(predictions, axis=1)

averaged_models = AveragingModels(models = (ENet, GBoost,
                                            KRR, lasso))
score = rmsle_cv(averaged_models)
print("Averaged base models score: {:.4f} ({:.4f})\n"
      .format(score.mean(), score.std()))


class StackingAveragedModels(BaseEstimator,
                             RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, _n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = _n_folds
        self.cloned_base_models = [list() for _model in self.base_models]
        self.cloned_meta_model = clone(self.meta_model)

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        # using self.cloned_base_models and cloned_meta_model
        kfold = KFold(n_splits=self.n_folds,
                      shuffle=True, random_state=69)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0],
                                            len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.cloned_base_models[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold
        # predictions as new feature
        self.cloned_meta_model.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and
    # use the averaged predictions as meta-features for the final
    # prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model
                             in base_models]).mean(axis=1)
            for base_models in self.cloned_base_models])
        return self.cloned_meta_model.predict(meta_features)

stacked_averaged_models = StackingAveragedModels(
    base_models=(ENet, GBoost, KRR), meta_model=lasso)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Average Modules score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))

model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))

model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))

'''RMSE on the entire Train data when averaging'''

print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*0.70 +
               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))

ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15

sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)