# import essentials
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

seed = 69 # Bill & Ted > Hitchhiker's Guide to the Galaxy
np.random.seed(seed)

train :pd.DataFrame = pd.read_pickle('../input/MoA/train_features.pkl')
target :pd.DataFrame = pd.read_pickle('../input/MoA/train_targets_scored.pkl')


train.drop(columns=['sig_id'],inplace=True)



data :pd.DataFrame = pd.concat([train,target],axis=1)
print(train.shape)
print(data.shape)
print(data.head())