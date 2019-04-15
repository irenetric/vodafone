# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:53:43 2019

@author: Pc_User
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

data = pd.read_csv('data_train.csv') #read the data

y = np.array(data['Footfall']) #get the variable to be predicted

data.drop('Footfall', axis = 1, inplace = True) #drop it from your dataset


n_features = data.shape[1] #get the number of features
n_obs = data.shape[0] #get the number of observations

feature_list = data.columns.values #get a list of all the features (coloumns header)

#an array of correlation coefficients between the targets and the data
corr_list = np.array([np.corrcoef(y, data[i])[0,1] for i in feature_list])


RSS_list = [] #collections of residual sum of squares - the lower the better
ESS_list = [] #collections of explained sum of squares - the higher the better

# now I use PCA, trying to reduce the number of features to the most relevant
# (orthogonal) ones. Since I can choose the number of features i can reduce the dataset 
# to, i am looping over all the possible 64 cases
# 
for i in range(1, n_features):
    reduced_data = PCA(n_components=i).fit_transform(data) #PCA with i features
    model = LinearRegression() #run a regression everytime
    model.fit(reduced_data, y)
    predictions = model.predict(reduced_data)
    RSS_list.append(np.sum((y - predictions)**2)) #append everytime the rss and ess
    ESS_list.append(np.sum((predictions - np.mean(predictions))**2))

best_RSS = (np.argmin(RSS_list), np.min(RSS_list)) #53, 1.270658
best_ESS = (np.argmax(ESS_list), np.max(ESS_list)) #54, 6.63

#use 53 regressors by PCA feature selection


        
        










