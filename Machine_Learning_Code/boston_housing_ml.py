#Predicting Boston Housing prices with machine learning
#Boston Housing data file in 'Data' folder
#Visualization files referenced in this code are in 'Visuals_Code' folder

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
%matplotlib inline

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)
print data.head(3)
print features.head(3)
print prices.head(3)