import os
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, roc_curve, auc

import torch
from torch import nn
from torchvision.models import resnet18
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
torchaudio.set_audio_backend("soundfile")

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

BASE_PATH = 'E:\\GravitationalWaveDetection\\'
SPLIT_RATIO = .9
RANDOM_STATE = 69
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
set_seed(RANDOM_STATE)

def apply_raw_path(row, is_train=True):
    file_name = row[0]
    if is_train:
        return os.path.join(
            BASE_PATH, 'train',
            file_name[0],
            file_name[1],
            file_name[2],
            file_name + ".npy")
    else:
        return os.path.join(
            BASE_PATH, 'test',
            file_name[0],
            file_name[1],
            file_name[2],
            file_name + ".npy")

df = pd.read_csv(os.path.join(BASE_PATH, 'training_labels.csv'))
# df = df.sample(frac=1, random_state=RANDOM_STATE)
df['file_path'] = df.apply(apply_raw_path, args=(True,), axis=1)
df['target'] = df['target'].astype('int8')

def plot_samples(sample_list):
    fig,a =  plt.subplots(len(sample_list), 3, figsize=(15,2 * len(sample_list)))
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=.5, wspace=0.2)
    cpt = 0
    for sample_num in sample_list:
        sample = np.load(df.loc[sample_num]['file_path'])
        a[cpt, 0].plot(sample[0],color='red')
        a[cpt, 1].plot(sample[1],color='green')
        a[cpt, 2].plot(sample[2],color='cyan')
        a[cpt, 1].set_title(
            'sample ' + str(sample_num) +
            ', label ' + str(df.loc[sample_num]['target']),
            fontsize=16
        )
        cpt = cpt + 1
    plt.show()
    return sample.shape

# samples = [0, 1, 2, 2547]
# plot_samples(samples)

df_train, df_val = train_test_split(
    df,
    train_size = SPLIT_RATIO,
    test_size = 1 - SPLIT_RATIO,
    random_state = RANDOM_STATE
)

print(len(df_train))
print(len(df_val))
print(f'-----------------\n' \
      f'{df_train.loc[0]}')

spectrogram = T.Spectrogram(n_fft=128, hop_length=64)

classes = torch.Tensor(df.target.unique())
model = resnet18(pretrained=True)
model.conv1=nn.Conv2d(1, model.conv1.out_channels,
                      kernel_size=model.conv1.kernel_size[0],
                      stride=model.conv1.stride[0],
                      padding=model.conv1.padding[0])
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))

train = df.sample(frac=1, random_state=RANDOM_STATE)
X = train.file_path
y = train.target
model.fit(torch.Tensor(np.load(X[0])), target[0])