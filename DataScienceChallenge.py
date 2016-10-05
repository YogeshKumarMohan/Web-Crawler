# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:17:22 2016

@author: Yogesh
"""

## Importing packages
import pandas as pd
from geopy.geocoders import GoogleV3
import math
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from dateutil import parser
import scipy.stats as stats
import seaborn as sns


# Question 1
Taxi_Green_Data = pd.read_csv("https://storage.googleapis.com/tlc-trip-data/2015/green_tripdata_2015-09.csv")

print("No. of Rows: %d \n No. of Columns: %d"%(Taxi_Green_Data.shape[0],Taxi_Green_Data.shape[1]))


# Question 2
# To calculate distance in miles
def distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    Source: http://gis.stackexchange.com/a/56589/15183
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    miles = 6367 * c*0.621371
    return miles

# Histogram showing distribution of Trip_distance
xbins = [x for x in range(0,20)]
x= Taxi_Green_Data['Trip_distance']
x.min()
x.max()

plt.hist(x, bins=xbins, color='green')
plt.axvline(x.mean(), color='b', linestyle='dashed', linewidth=2, label = 'Mean')
plt.axvline(x.median(), color='b', linewidth=2, label ='Median')
plt.title("Histogram for the distribution of trip distance")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.legend()    
plt.show()


# Question 3:
# part1
pd1 = pd.DataFrame(Taxi_Green_Data)
pd1['Day_Hour'] = pd.Series([parser.parse(pd1.iloc[x]['lpep_pickup_datetime']).hour for x in range(len(pd1))])

grouped = pd1.groupby('Day_Hour')
Mean_Median_perhour = grouped.agg(['mean','median'])['Trip_distance']
print("Reporting mean and median trip distance grouped by hour of day:")
print(Mean_Median_perhour)


#part2

# NYC Airport Area Taxi based on RateCodeID

grouped1 = pd1[pd1.RateCodeID.isin([2,3])].groupby('RateCodeID')
grouped1.agg(['count','mean','max','min'])['Total_amount']
print("RateCodeID : 2 = JFK Airport Area \n RateCodeID : 3 Newark Airport Area")

# Additional interesting characteristics of these trips
(pd1[pd1.RateCodeID == 2].Total_amount<0).value_counts()
pd2 = pd1[pd1.RateCodeID.isin([2,3])]
group3 = pd2[pd2.Total_amount <= 0].groupby(['Payment_type','RateCodeID'])
group3.agg('count')['Total_amount']

print("RateCodeID : 2 = JFK Airport Area ---  RateCodeID : 3 Newark Airport Area")
print("Payment_type : 2 = Cash ---  Payment_type : 3 = No Charge ---  Payment_type : 4 = Dispute ")


#Q4

#Part1
# Creating new  derived Variable Tip_Percentage
pd1['Tips_Percentage']= pd.Series([(pd1.iloc[x]['Tip_amount']/pd1.iloc[x]['Total_amount'])*100 for x in range(len(pd1))])

group4 = pd1[pd1['Tip_amount']>0].groupby('Payment_type')
No_of_non_zeroTransaction = group4.agg('count')['Total_amount']

print("Filtering only data points with payment type Credit card:")
pd2 = pd1[pd1.Payment_type == 1]

print("Distribution of Tips_Percentage:")
pd2.hist('Tips_Percentage')

print("Correlation Matrix of features")
cols=['Passenger_count','Trip_distance', 'Fare_amount','Tip_amount','Total_amount', 'Day_Hour']
cm = np.corrcoef(pd2[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},yticklabels=cols, xticklabels=cols)
plt.show()

# Importing required packages for building the predictive model
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Dropping irrelevent features
pd3 = pd2.drop(pd2.columns[[1,2,3,17,16]],axis=1)
# Dropping missing values
pd3= pd3.dropna()


X = pd3.iloc[:, :-1].dropna().values
y = pd3['Tips_Percentage'].dropna().values

# Splitting Training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)
# Training the model with Training set
forest = RandomForestRegressor( n_estimators=100,criterion='mse',random_state=1,n_jobs=-1)
forest.fit(X_train, y_train)

# Validating using prediction
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)


# Validation using metircs
print('MSE train: %.3f, test: %.3f' % (
mean_squared_error(y_train, y_train_pred), 
mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.3f, test: %.3f' % ( r2_score(y_train, y_train_pred), 
                                       r2_score(y_test, y_test_pred)))
                                       
# MSE train: 0.003, test: 0.062
# R^2 train: 1.000, test: 0.999
                                       

# Plot for feature importance of the predictive model
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


#Q5
# New derived variable Tripduration
pd1['Trip_duration'] = pd.Series([(parser.parse(pd1.iloc[x]['Lpep_dropoff_datetime'])-parser.parse(pd1.iloc[x]['lpep_pickup_datetime'])).total_seconds() for x in range(len(pd1))])

#New derived variable Average_speed
pd1['Average_Speed'] = pd.Series([(pd1.iloc[x]['Trip_distance']/pd1.iloc[x]['Trip_duration'])*3600 for x in range(len(pd1))])

#Week1 - 1 to 6
#Week2 - 7 to 13
#Week3 - 14 to 20
#Week4 - 21 to 27
#Week5 - 28 to 30

# New derived variable creation based on Week the trip happened
Week_no = []

for x in range(len(pd1)):
    if parser.parse(pd1.iloc[x]['lpep_pickup_datetime']).day in range(1,7):
        Week_no.append(1)
    elif parser.parse(pd1.iloc[x]['lpep_pickup_datetime']).day in range(7,14):
        Week_no.append(2)
    elif parser.parse(pd1.iloc[x]['lpep_pickup_datetime']).day in range(14,21):
        Week_no.append(3)
    elif parser.parse(pd1.iloc[x]['lpep_pickup_datetime']).day in range(21,28):
        Week_no.append(4)
    elif parser.parse(pd1.iloc[x]['lpep_pickup_datetime']).day in range(28,31):
        Week_no.append(5)
    else:
        Week_no.aapend(0)

pd1['Week_no']= pd.Series(Week_no)

# Removing outliers
pd5 = pd1[pd1.Average_Speed<100]

group7 = pd5.groupby('Week_no')
group7.agg('mean')['Average_Speed']

# Importing stas package with Anova funcion
from scipy import stats

week1 = pd5[pd5['Week_no']==1]['Average_Speed']
week2 = pd5[pd5['Week_no']==2]['Average_Speed']
week3 = pd5[pd5['Week_no']==3]['Average_Speed']
week4 = pd5[pd5['Week_no']==4]['Average_Speed']
week5 = pd5[pd5['Week_no']==5]['Average_Speed']
#(array([ 30.28285342]), array([  3.28674610e-25]))

week1 = week1.sample(10000)
week2 = week2.sample(10000)
week3 = week3.sample(10000)
week4 = week4.sample(10000)
week5 = week5.sample(10000)

f_val, p_val = stats.f_oneway(week1, week2, week3,week4, week5)

print("Results for Anova Test")
print("f-value: %f"%(f_val))
print("p-value: %f"%(p_val))


week1 = pd5[pd5['Week_no']==1]['Trip_distance']
week2 = pd5[pd5['Week_no']==2]['Trip_distance']
week3 = pd5[pd5['Week_no']==3]['Trip_distance']
week4 = pd5[pd5['Week_no']==4]['Trip_distance']
week5 = pd5[pd5['Week_no']==5]['Trip_distance']
#(array([ 12.02065077]), array([  9.16788359e-10]))

week1 = week1.sample(10000)
week2 = week2.sample(10000)
week3 = week3.sample(10000)
week4 = week4.sample(10000)
week5 = week5.sample(10000)

f_val, p_val = stats.f_oneway(week1, week2, week3,week4, week5)

print("Results for Anova Test")
print("f-value: %f"%(f_val))
print("p-value: %f"%(p_val))



    










