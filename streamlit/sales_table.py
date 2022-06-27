import pandas as pd
import streamlit as st

df = pd.read_csv('sales_table.csv')
df['create_date'] = pd.to_datetime(df['create_date'])
st.dataframe(df.iloc[:-20:-1])
month_data = df.groupby(by=[df.create_date.dt.month])

st.line_chart(month_data['profit'].sum())