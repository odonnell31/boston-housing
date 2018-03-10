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
#%matplotlib inline

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)
print data.head(3)
print features.head(3)
print prices.head(3)
print "======"

## Some EDA...

# Minimum price of the data
minimum_price = np.amin(prices)
# Maximum price of the data
maximum_price = np.amax(prices)
# TODO: Mean price of the data
mean_price = np.mean(prices)
# Median price of the data
median_price = np.median(prices)
# Standard deviation of prices of the data
std_price = np.std(prices)

# Show the calculated statistics
print "Statistics for Boston housing dataset:\n"
print "Minimum price: ${:,.2f}".format(minimum_price)
print "Maximum price: ${:,.2f}".format(maximum_price)
print "Mean price: ${:,.2f}".format(mean_price)
print "Median price ${:,.2f}".format(median_price)
print "Standard deviation of prices: ${:,.2f}".format(std_price)
print "====="

# Add a performance metric function
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)
    
    # Return the score
    return score

# Test the performance metric function
#score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
#print "TEST: Model has a coefficient of determination, R^2, of {:.3f}.".format(score)
    
# Import 'train_test_split' to split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size = 0.2, random_state = 31)
# Success
print "Training and testing split was successful."
print "====="



