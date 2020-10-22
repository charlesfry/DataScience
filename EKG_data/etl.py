import numpy as np
import pandas as pd
import wfdb
import ast


def load_raw_data(df,sampling_rate,path) :
    if sampling_rate == 100 :
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else :
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal,meta in data])
    return data

path = 'E:\DataScience\ptb_xl\\'
sampling_rate = 100

# load and convert annotation data
Y = pd.read_csv(path + 'ptbxl_database.csv',index_col='ecg_id')

Y.scp_codes = Y.scp_codes.apply(
    lambda x: ast.literal_eval(x)
)


# load raw signal data
X = load_raw_data(df=Y,sampling_rate=sampling_rate,path=path)

# load scp_statements.csv ofr diagnostic aggregation
agg_df = pd.read_csv(path + 'scp_statements.csv',index_col=0)
print(agg_df.head())
print(f'\n{agg_df.keys()}')
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic:pd.DataFrame) :
    tmp = []
    for key in y_dic.keys() :
        if key in agg_df.index :
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# split into train/test
test_fold = 10

# train
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[Y.strat_fold != test_fold].diagnostic_superclass

# test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

print(f'{y_train}\n\n')
print(f'{y_train.shape}\n\n')

for val in pd.Series(y_train) :
    if val: print(val)

# Run tests
scp = pd.read_csv(f'{path}/scp_statements.csv')
