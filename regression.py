# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:53:43 2019

@author: Pc_User
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.regression.linear_model import OLS

datac = pd.read_csv('data_train.csv') #read the data

y = np.array(datac['Footfall']) #get the variable to be predicted

datac.drop(['Footfall'], axis = 1, inplace = True) #drop from your dataset

n_features = datac.shape[1] #get the number of features
n_obs = datac.shape[0] #get the number of observations

feature_list = datac.columns.values #get a list of all the features (coloumns header)






dummy_tipologia = pd.DataFrame(pd.get_dummies(datac['Tipologia_encoded']))
dummy_provincia = pd.DataFrame(pd.get_dummies(datac['Provincia_encoded']))
dummy_localita = pd.DataFrame(pd.get_dummies(datac['Localita_encoded']))
dummy_area = pd.DataFrame(pd.get_dummies(datac['area_encoded']))


data = pd.concat([datac, dummy_tipologia, dummy_provincia, dummy_localita], axis = 1, ignore_index = True)

###############   Linearity test - Ramsey RESET    #############
model = LinearRegression()
model.fit(data, y)
y_hat = model.predict(data) #get y hat

y_squared = pd.DataFrame(y**2)
y_third = pd.DataFrame(y**3)

merged_dataset = pd.concat([data, y_squared, y_third], axis = 1)

modelu = LinearRegression()
modelu.fit(merged_dataset, y)
y_merged = modelu.predict(merged_dataset)

RSS_restr = np.sum((y_hat - y)**2)
RSS_unr = np.sum((y_merged - y)**2)
j = 2 
F_test = ((RSS_restr - RSS_unr) / j) / (RSS_unr / (n_obs - (n_features + 2) - 1))

#n_features = data.shape[1] #get the number of features
#n_obs = data.shape[0] #get the number of observations
#
#feature_list = data.columns.values #get a list of all the features (coloumns header)
#
##an array of correlation coefficients between the targets and the data
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


