# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:53:43 2019

@author: Pc_User
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA


df = pd.read_csv('data_train.csv') #read the data

y = np.array(df['Footfall']) #get the variable to be predicted

df.drop(['Footfall'], axis = 1, inplace = True) #drop from your dataset
df.drop(['Store_ID'], axis = 1, inplace = True) #also this is just irrelevant

n_features = df.shape[1] #get the number of features
n_obs = df.shape[0] #get the number of observations

feature_list = df.columns.values #get a list of all the features (coloumns header)

#get dummies for each categorical variables
dummy_tipologia = pd.DataFrame(pd.get_dummies(df['Tipologia_encoded']))
dummy_provincia = pd.DataFrame(pd.get_dummies(df['Provincia_encoded']))
dummy_localita = pd.DataFrame(pd.get_dummies(df['Localita_encoded']))
dummy_area = pd.DataFrame(pd.get_dummies(df['area_encoded']))

#attach to the dataset all the dummy variables
data = pd.concat([df, dummy_tipologia, dummy_provincia, dummy_localita], axis = 1, ignore_index = True)

#check if the model can be fitted linearly
###############   Linearity test - Ramsey RESET    #############
model = LinearRegression()
datadf = pd.DataFrame(PCA(n_components=5).fit_transform(data))
model.fit(datadf, y)
y_hat = model.predict(datadf) #get y hat

y_squared = pd.DataFrame(y_hat**2)
y_third = pd.DataFrame(y_hat**3)

merged_df = pd.concat([datadf, y_squared, y_third], axis = 1)


modelu = LinearRegression()
modelu.fit(merged_df, y)
y_merged = modelu.predict(merged_df)

RSS_restr = np.sum((y_hat - y)**2)
RSS_unr = np.sum((y_merged - y)**2)
j = 2
F_test = ((RSS_restr - RSS_unr) / j) / (RSS_unr / (n_obs - 7 - 1))

cv = 4.667 #df(2, 300) at a significance level 0.01

####F_test < critical value, no evidence to reject the null hypothesis that the
###higher order parameter are equal to zero.


##create array of correlation coefficients between the targets and the data
dat = np.array(data)
corr_list = []
for i in range(len(data.columns.values)):
    corr_list.append(np.corrcoef(y, dat[:,i])[0,1])
#
#
RSS_list = [] #collections of residual sum of squares - the lower the better
#
#
#for i in range(1, data.shape[1]):
#    reduced_data = PCA(n_components=i).fit_transform(data) #PCA with i features
#    model = LinearRegression() #run a regression everytime
#    model.fit(reduced_data, y)
#    predictions = model.predict(reduced_data)
#    RSS_list.append(np.sum((y - predictions)**2)) #append everytime the rss and ess
    

    
#
#    
#best_RSS = (np.argmin(RSS_list), np.min(RSS_list)) #291, 0.140658
reduced_data = PCA(n_components=200).fit_transform(data) #PCA with i features
model = LinearRegression() #run a regression everytime
model.fit(reduced_data, y)
predictions = model.predict(reduced_data)
RSS_list.append(np.sum((y - predictions)**2)) #append everytime the rss and ess


reduced_data = np.array(reduced_data)
#corr_listr = []
#for i in range(reduced_data.shape[1]):
#    corr_listr.append(np.corrcoef(y, reduced_dat[:,i])[0,1])
#
#for j in corr_list:
#    print(j in corr_listr)
#

data = np.array(data)
dimensions_kept = []
for j in range(data.shape[1]):
    for i in range(reduced_data.shape[1]):
        if np.all(data[:, j] == reduced_data[:, i]):
            dimensions_kept.append(j)
#            
#coefficient_significance = []
#for i in range(model.coef_.size):
#    coefficient_significance.append(model.coef_[i]/ np.sqrt(np.var(model.coef_[i])))
#


