# UdacityCapstone
Starbucks Capstone project for Udacity

run the following command in your command prompt prior to running this iPython notebook:
pip install -r requirements.txt 

This will download a compatible version of:
  pandas, the dataframe software we use
  sklearn-pandas, which we use to map functions to pandas
  and category encoders, which are some of the functions that we use on our dataframes

then you can run the .ipynb file

# Motivation for the project
Starbucks has an app that gives users targeted ads. They want to target their ads to maximize the number of people who make purchases
in their store in order to maximize their profits. But, they do not want to give ads to people who would have already made purchases,
as this would give their money away. Starbucks has given us several types of ads, and a randomized dataset of individuals who received
these ads. We have created a model that predicts consumer behavior when given a particular ad with 76.8% accuracy. Other papers 
(which we will not name out of politeness) reported higher accuracy, but their methodology was flawed. Our model correctly calculates the 
Reward category by reporting the potential reward, not the reward accumulated in the transaction, which would be unknown in real test data.

Our report highlights differences in consumer behavior over several demographic vectors. Notably, we found that our consumers preferred 
coupons with lower difficulty and reward levels, and that high income individuals were more apt to complete coupons than low income individuals.