import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from functools import reduce
from collections import Counter

# st.set_page_config(layout='wide')

# ETL
FILEPATH = './data.csv'
@st.cache
def etl(filepath, strip_stars=False):
    df = pd.read_csv(filepath).fillna('')
    if strip_stars:
        return df.applymap(lambda x: x[:-1] if x.endswith('*') else x)
    return df
df = etl(FILEPATH, strip_stars=False)
clean_df = etl(FILEPATH, strip_stars=True)
names = sorted(set(clean_df.to_numpy().flatten()))
names.remove('')

###
st.title("AVERAGE NUMBER OF PEOPLE AT STANDUP")
@st.cache
def get_average_num(df:pd.DataFrame):
    li = []
    for row in df.to_numpy():
        n = len(list(filter(lambda x: len(x) > 0, row)))
        li.append(n)
    return round(np.array(li).mean(),2)
avg_num_in_standup = get_average_num(df)
st.write('There were **{:.2f}** people at standup on average'.format(avg_num_in_standup))
###

st.title("NUMBER OF TIMES EACH PERSON ATTENDED STANDUP")
st.write('This counts both in-person and remote attendance.')
# @st.cache
def get_counts(df:pd.DataFrame):
    li = df.to_numpy().flatten()
    li = filter(lambda x: x != '', li)
    c = Counter(li)
    c = dict(sorted(c.items(), key=lambda item: item[1], reverse=True))
    return pd.DataFrame(list(c.items()), columns=['Name', 'Attendance']).set_index('Name')
st.table(get_counts(clean_df))
st.write('Note, because Charles took all the data, he was necessarially there every day that the data was taken.')

st.title('HOW MANY TIMES DID PEOPLE CALL IN TO STANDUP REMOTELY?')
@st.cache
def get_wfh_counts(data:pd.DataFrame):
    df = data.copy()
    li = filter(lambda x: '*' in x,df.to_numpy().flatten())
    li = map(lambda x: x[:-1], li)
    c = Counter(li)
    return dict(sorted(c.items(), key=lambda item: item[1]))
wfh_counts = get_wfh_counts(df)
st.table(pd.DataFrame(list(wfh_counts.items()), columns=['Names', 'Times called in']).set_index('Names'))
st.write('David is the most dedicated to coming in. No surpise there. Also, most of the times that Diego called in were because he was stuck in traffic.')
###

st.title('WHO WAS THE MOST POPULAR?')
st.write("""This section calculates what your average pick order was. The lower, the more quickly you were called.
Therefore, if you were 'popular', you were picked closer to the beginning. Note, this only counts if you were not first that day, since nobody 'picked' you to go first.""")
@st.cache
def get_vals(name:str, df):
    data = (clean_df == name ) * np.arange(df.shape[1])
    avg = data.sum(axis=1)
    avg = avg.where(lambda x: x > 0)
    avg.dropna(inplace=True)
    mean = avg.mean()
    mean = '{:.2f}'.format(round(mean, 2))
    return mean
popular = {k:get_vals(k, clean_df) for k in names}
popular = dict(sorted(popular.items(), key=lambda item: item[1]))
st.table(pd.DataFrame(list(popular.items()), columns=['Name', 'Average Turn Order']).set_index('Name'))
st.write('Note: Alex was the least popular because he was only picked once due to being first every day except one. Still counts.')

###

st.title("DID WORKING REMOTELY AFFECT YOUR PICK ORDER?")

def get_remote_totals(df):
    total = (df.applymap(lambda x: True if x != '' else False) * np.arange(1, df.shape[1] +1)).max(axis=1)
    local_np = (df.applymap(lambda x: len(x) > 0 and not x.endswith('*')) * np.arange(1, df.shape[1] + 1)).to_numpy().flatten()
    local_avg = local_np[local_np > 0].mean()
    remote_np = (df.applymap(lambda x: len(x) > 0 and x.endswith('*')) * np.arange(1, df.shape[1] + 1)).to_numpy().flatten()
    remote_avg = remote_np[remote_np > 0].mean()
    office_avg = total.mean()
    return [
    'If you are local, your average turn number is: **{:.1f}**'.format(local_avg)
    ,'If you are remote, your average turn number is: **{:.1f}**'.format(remote_avg)
    ,'If you are remote, you will be picked **{:.1f}** people later than if you work in the office.'.format(remote_avg - local_avg)
    ,'AKA if you are remote, you will be picked **{:.1f}%** later than if you work in the office.'.format(100 * (remote_avg - local_avg) / office_avg)
    ]
for line in get_remote_totals(df):
    st.write(line)

###

st.title('And now the best part: everyone\'s favorite people!')
st.title('This section tells you who picked you the most!')
def get_top_favorites(data:pd.DataFrame, name:str):
    c = Counter()
    for li in data.to_numpy():
        indicies = filter(lambda x: li[x + 1] == name, range(len(li)-1))
        names = [li[i] for i in indicies]
        c.update(Counter(names))
    f = {k:v for k,v in c.items() if v == c.most_common()[0][1]}
    return list(f.keys()), c.most_common()[0][1]

def favorites(df):
    favorite_persons = {k:get_top_favorites(df, k) for k in names}
    favorite_persons = dict(sorted(favorite_persons.items(), key=lambda item: item[0], reverse=False))
    li = []
    for k,v in favorite_persons.items():
        suffix = 's!' if v[1] > 1 else '!'
        li.append(f'**{k}** was picked by **{"** and **".join(v[0])}** **{v[1]}** time' + suffix)
    return li
li = favorites(clean_df)
for l in li:
    st.write(l)

###
st.title("THAT'S (mostly) ALL, FOLKS!")
st.write('Breakdown of everyone\'s picks')
st.write("These tables count how many times each person picked someone")
def get_all_favorites(data:pd.DataFrame, name:str):
    c = Counter()
    for li in data.to_numpy():
        indicies = filter(lambda x: li[x - 1] == name, range(1,len(li)))
        names = [li[i] for i in indicies]
        c.update(Counter(names))
        del c['']
    c = dict(sorted(c.items(), key=lambda item: item[1], reverse=True))
    return pd.DataFrame(list(c.items()), columns=['Name', 'Number of times picked by ' + name]).set_index('Name')
for n in names:
    st.write(f'**{n}**')
    st.table(get_all_favorites(clean_df, n))

st.title('And the database itself in case any of you want to work with it:')
st.write('note, \'*\' means that the person called in remotely.')
st.dataframe(df)
csv = df.to_csv().encode('utf-8')
st.download_button(label='Download the data', data=csv, file_name='standup.csv')