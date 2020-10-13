# extract, transform, and load the data

# first, import our essentials
import numpy as np
import pandas as pd
import os
import pickle

# get our data
def extract(dirs_path) :
    dataset_paths = []
    for root,dirs,files in os.walk(dirs_path) :
        for name in files :
            file = os.path.join(root,name)
            dataset_paths.append(file)
    return dataset_paths

paths = extract('./input/')

# check that all datasets have the same keys
def check_keys(paths) :
    for i in range(1,len(paths)) :
        df_1 = pd.read_csv(paths[i-1])
        df_2 = pd.read_csv(paths[i])
        if not np.array_equal(df_1.keys(),df_2.keys()) :
            print(i)
            return False
    return True

print(check_keys(paths)) # returns true

