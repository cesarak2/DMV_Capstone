import numpy as np
import pandas as pd
from cgi import print_environ_usage
import os
from os.path import exists # check if SQL exists
#import re
#import smtplib, ssl
#from subprocess import CalledProcessError
#import time
#from datetime import datetime
#import random #for random refresh times
from pprint import pprint
import sqlite3
import sqlite3 as db
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sqlalchemy import outerjoin
# model training
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import base
from sklearn.linear_model import LinearRegression, SGDRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
from sklearn import base
# hyperparameters
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# model evaluation
from sklearn.metrics import mean_squared_error
import math


# # # # # # # # # # # # # # # # # # # # # # # # # # #
#       MAKING AND TESTING THE SQL CONNECTION       #
# # # # # # # # # # # # # # # # # # # # # # # # # # #
connection = sqlite3.connect('DB_DMV.sqlite')
# testing a simple query:
query = """
    SELECT * FROM REAL_ID LIMIT 10;
    """
c = connection.cursor()
r = c.execute(query)
a = r.fetchall() # returns [(48031, 17586)]
#https://www.knowledgehut.com/tutorials/python-tutorial/python-database-connectivity


# # # # # # # # # # # # # # # # # # # # # # # # ##
# GETS ALL SERVICES BY EXTRATING TABLES FROM SQL #
# # # # # # # # # # # # # # # # # # # # # # # # ##
query = "SELECT name FROM sqlite_schema WHERE type='table' ORDER BY name;"
c.execute(query) #executes query
list_of_services = [] #defines empty list to be populated
while True: # loops through all tables and retrieve their names
   record=c.fetchone()
   if record==None: # breaks when there is no more data to fetch
       break
   list_of_services.append(record[0])

if 'dmv_zips' in list_of_services: # removes later additions to the database
    list_of_services.remove('dmv_zips')
    print('contains')
print('List of services: '+ str(list_of_services))
'''
# returns List of services: ['CDL_PERMIT_OR_ENDORSEMENT', 'INITIAL_PERMIT',
#   'KNOWLEDGE_TESTING', 'NEW_TITLE_OR_REGISTRATION', 'NONDRIVER_ID',
#   'REAL_ID', 'REGISTRATION_RENEWAL', 'RENEWAL_CDL',
#   'RENEWAL_LICENSE_OR_NONDRIVER_ID', 'TITLE_DUPLICATE_or_REPLACEMENT',
#   TRANSFER_FROM_OUT_OF_STATE']
'''


# # # # # # # # # # # # # # # # # # # # # # # # # ##
# GETS ALL CITIES BY EXTRATING COLUMNS SQL COLUMNS #
# # # # # # # # # # # # # # # # # # # # # # # # # ##
query = "PRAGMA table_info(" + list_of_services[0] + ");" #uses 1st service as baseline
c.execute(query)
list_of_locations = [] #defines empty list to be populated
while True: # loops through all tables and retrieve their column names, i.e., locations
   record=c.fetchone()
   if record==None: # breaks when there is no more data to fetch
       break
   list_of_locations.append(record[1])
print('List of locations: ' + str(list_of_locations))
'''
# returns List of locations: ['time_stamp', 'Bakers_Basin', 'Bayonne',
#   'Camden', 'Cherry_Hill', 'Cardiff', 'Delanco', 'East_Orange',
#   'Eatontown', 'Edison', 'Elizabeth', 'Flemington', 'Freehold',
#   'Hazlet', 'Jersey_City', 'Lakewood', 'Lodi', 'Manahawkin',
#   'Medford', 'Newark', 'Newton', 'North_Bergen', 'Oakland',
#   'Paterson', 'Rahway', 'Randolph', 'Rio_Grande', 'Runnemede',
#   'Salem', 'Somerville', 'South_Brunswick', 'South_Plainfiled',
#   'Springfield', 'Toms_River', 'Trenton_Regional', 'Turnersville',
#   'Vineland', 'Wallington', 'Washington', 'Wayne', 'West_Deptford']
'''

# # # # # # # # # # # # # # # # # # # # # # # # # # #
#            COMPARES NULLS AGAINST VALUES          #
# # # # # # # # # # # # # # # # # # # # # # # # # # #
# Checks is the NULLS are text or types:
query = """
SELECT SUM(CASE when Bayonne is NULL then 1 else 0 end) count_nulls 
, count(Bakers_Basin) count_not_nulls from INITIAL_PERMIT
"""
c.execute(query)
c.fetchall() #if count_nulls is 0, NULLS are stores as text.

'''
# # # # # # # # # # # # # # # # # # # # # # # # # ##
#            REPLACES 'NULL' WITH NULL             #
# # # # # # # # # # # # # # # # # # # # # # # # # ##
# The file originally had NULL values stores as text.
# This sections transforms all 'NULL' into NULL
# TRANSFORMS 'NULL' INTO NULL
for service in list_of_services:
    for city in list_of_locations:
        print(service, city)
        # for each service, updates each city
        query=""" 
        UPDATE 
            """ + service + """
        SET """ + \
            city + """ = null WHERE """ + city + """ = 'NULL'
        """
        c.execute(query)
    print(service)
connection.commit()
'''

# # # # # # # # # # # # # # # # # # 
#         CONVERTING TO DF        #
# # # # # # # # # # # # # # # # # # 
service = list_of_services[5] # !! CHANGE HERE FOR DIFFERENT SERVICES !!
query = """
    SELECT * FROM """ + service
c = connection.cursor()
raw_service = pd.read_sql(query, connection)
raw_service.shape #(57003, 41)

# # # # # # # # # # # # # # # # # # # # # # # #
#            FIXING THE MISSING AM PM         #
# # # # # # # # # # # # # # # # # # # # # # # #
# 1) if day is the same and time is smaller than previous, shift = pm
# 2) if day has a pm, the missing ones are am
#    most of dates should have a resolution by now
# 3) if day is the very next to the previous one and hour diff is small = am
# 4) if day is the very previous to the next one and hour diff is small = pm
raw_service['shift'] = '' #calls everyone am and change the ones repeated to pm

raw_service['time_stamp'] = pd.to_datetime(raw_service['time_stamp'], format='%d/%m/%Y %H:%M')
raw_service['time_shift'] = raw_service['time_stamp'].shift(1)
# 1) gives first pm of day (time drops to 0 but day stays the same)
#   df['shift'] = ''
raw_service['shift'] = raw_service.apply(lambda x: 'pm' if \
                (x.time_stamp + pd.offsets.Hour(12)).hour < (x.time_shift + pd.offsets.Hour(12)).hour and \
                x.time_stamp.day == x.time_shift.day \
                else 'am', axis=1)

# 1.1) completes pm for all values of the day
for i in range(1, len(raw_service)):
    # if previous is pm and current is the same day than previous, it is also pm
    if raw_service['shift'][i-1] == 'pm' and raw_service['time_stamp'][i].day == raw_service['time_stamp'][i-1].day:
        raw_service['shift'].iloc[i] = 'pm'
raw_service.groupby('shift').count() # used to verify if any blank value is present

'''
4 cases (old -> fixed)
am < 12 -> do nothing (11:15 -> 11:15)
pm = 12 -> do nothing (noon = 12:15 -> 12:15)
am = 12 -> subtract 12h (midnight fifteen = 12:15 -> 00:15)
pm < 12 -> adds 12 hours (11:00 -> 23:00) 
'''
raw_service['time_shift'] = raw_service.apply(lambda x: x['time_stamp'] if (x['shift'] == 'am' and x['time_stamp'].hour < 12) or \
                    (x['shift'] == 'pm' and x['time_stamp'].hour == 12) else \
                        x['time_stamp'] - pd.Timedelta(hours=12) if x['shift'] == 'am' and x['time_stamp'].hour == 12 else \
                            x['time_stamp'] + pd.Timedelta(hours=12), axis=1)
raw_service.drop(['time_stamp', 'shift'], axis=1, inplace = True)

cols = raw_service.columns.tolist() #change columns order...
cols = cols[-1:] + cols[:-1] 
raw_service = raw_service[cols] # ...to bring new time to the first position

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  DROPPING THE COLUMNS THAT DON'T HAVE ANY APPOINTMENTS  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# identify and remove columns that don't have any appointment
total_entries = raw_service.shape[0] # number of rows
list_to_drop = [] # for information
columns = raw_service.columns
for column in columns:
    if raw_service.loc[:, column].isna().sum() == total_entries: # all entries are NAs
        list_to_drop.append(column)
        raw_service.drop([column], axis = 1, inplace=True) #drop columns
    elif raw_service[column][-1:] is not None: # there are appointments, and the last is valid
        pass
    else: #to avoid leaking appointment from next agency, create a fake avent at the last position
        raw_service[column][-1:] = raw_service['time_shift'][-1:]



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  CREATING FAKE APPOINTMENTS BEFORE TIME GAPS, AVOIDING OUTLIERS #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# the data was captured in different time periods, so to avoid a big gap between them, a fake event
# is created at the end of each time period. Ideally this step is be avoided by using a better
# data collection.
# Assumptions: (1) all locations without any appointment were already removed, (2) if that agency
# has enough appointments, this extra event will not change much, and (3) if that agency has seldom
# events, this approach will reduce the variability of results (by bringing mean down).

#list_to_drop = []
columns = raw_service.columns
for i in range(0, len(raw_service)-1): # for every entry
    if (raw_service['time_shift'][i+1] - raw_service['time_shift'][i]).days > 2: # if more than given days between time and next
        for column in columns: # for that time gap, get every location
            if raw_service[column][i] is None: # last value is not valid
                raw_service[column][i] = raw_service['time_shift'][i].strftime('%m/%d/%Y %I:%M %p') # 
        print(str(raw_service['time_shift'][i-1]) + ' - ' + str(raw_service['time_shift'][i]) + ': bigger than given difference in days')
# returns 2021-10-26 18:48; 2021-11-13 12:59; 2021-12-03 20:55; 2022-02-18 02:25
# as the final values before gaps bigger than two days
#raw_service.to_pickle(service + '_df_raw') #saves finished df for analysis

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  CONVERTS HOUR df INTO VERTICAL df (MELT) FOR EASIER ACCESS #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
service_df = pd.melt(raw_service, id_vars=['time_shift'], value_vars=raw_service.columns[1:], var_name = 'location') # melts original df
service_df['hasAvailability'] = service_df['value'].apply(lambda x: 0 if x is None else 1) # 1 if that time is available for that service


# # # # # # # # # # # # # # # # # # # # # # # #
#           GETS NEXT AVAILABLE TIME          #
# # # # # # # # # # # # # # # # # # # # # # # #
service_df['value'] = pd.to_datetime(service_df['value'], format='%m/%d/%Y %I:%M %p') #directly from the DMV website
service_df['time_shift'] = pd.to_datetime(service_df['time_shift'], format='%Y-%m-%d %H:%M')

service_df['diffOfTime'] = service_df['value'] - service_df['time_shift'] # horizontal distance between times
#service_df['diffTimeDays'] = service_df['diffOfTime'].apply(lambda x: x.days) # horizontal distance between times in days
# vertical distance = the n of intervals until next available appointment
service_df['countdown_n'] = service_df.groupby((service_df['hasAvailability'] == 1).cumsum()).cumcount(ascending=False)+1
service_df.loc[service_df['hasAvailability'] == 1, 'countdown_n'] = 0


# # # # # # # # # # # # # # # # # # # # # # # # #
#           GETS NEXT APPOINTMENT TIME          #
# # # # # # # # # # # # # # # # # # # # # # # # #
service_df['temp_next_time'] = 0
for i in range(len(service_df)):
    try:
        service_df['temp_next_time'][i] = service_df.iloc[i + service_df['countdown_n'][i], 0]
    except IndexError:
        service_df['temp_next_time'][i] = '0'
service_df['temp_next_time'] = pd.to_datetime(service_df['temp_next_time'], format='%Y-%m-%d %H:%M') #directly from the DMV website
service_df['diffOfTime_vertical'] = service_df['temp_next_time'] - service_df['time_shift'] # horizontal distance between times


# cleans df for clarity, dropping temporaty columns
service_df_cleaned = service_df.copy()
service_df_cleaned.drop(['value', 'hasAvailability', 'diffOfTime', 'countdown_n', 'temp_next_time'], axis = 1, inplace=True)
service_df_cleaned.columns


# service_df['diffOfTime_vertical'].plot(ylim=[0, .1E15]) #checks how times are distributed

'''
# # # # # # # # # # # # # # # # # # # # # # # # #
#          FIND FREQUENCY BY USING FFT          #
# # # # # # # # # # # # # # # # # # # # # # # # #
# Try to find a temporal frequency appliying fast Fourier transform.
valid_locations = service_df.location.unique()
import scipy.fft
for location in valid_locations:
    data_solo_location = service_df.groupby('location').get_group(location) #splits dataset per city
    diff_vertical_in_s = np.array(data_solo_location.diffOfTime_vertical/ np.timedelta64(1, 's'))
    fft = scipy.fft.fft((diff_vertical_in_s - diff_vertical_in_s.mean()))
    plt.plot(np.abs(fft))
    plt.title('FFT for ' + str(location))
    plt.show()
# no valid frequency exists, hence this method can't be applied.
'''


# # # # # # # # # # # # # # # # # # # # # # # # #
#      USING TIME AS CATEGORICAL VARIABLES      #
# # # # # # # # # # # # # # # # # # # # # # # # #
service_df_cleaned['days_cat'] = service_df_cleaned.apply(lambda x: x.time_shift.weekday(), axis=1)
service_df_cleaned['hours_cat'] = service_df_cleaned.apply(lambda x: x.time_shift.hour, axis=1)
service_df_cleaned['minutes_cat'] = service_df_cleaned.apply(lambda x: x.time_shift.minute, axis=1)
#service_df_cleaned.to_pickle(service + '_df_cleaned') #saves finished df for analysis

#plt.plot(service_df_cleaned.time_shift, service_df_cleaned.diffOfTime_vertical)
#service_df_cleaned.diffOfTime_vertical.idxmin()
#service_df_cleaned[113975:113985]
# # # # # # # # # # # # # # # # # # # # # # # # # # #
#                  MODEL TRAINING                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # #
class cat_estimator(base.BaseEstimator, base.RegressorMixin):
    def __init__(self, column, estimator_factory):
        # column is the value to group by; estimator_factory can be
        # called to produce estimators
        self.column = column
        self.estimator_factory = estimator_factory
        
    def fit(self, X, y):
        # Create an estimator and fit it with the portion in each group
        # uses group_by split df by locations. Train a model per lcoation
        # and store them in a dictionary
        self.ests = {}
        groups = X.groupby(self.column).groups #splits by location
        for group, rows in groups.items():
            self.ests[group] = self.estimator_factory() #calls factory per model
            self.ests[group].fit(X.loc[rows], y.loc[rows]) #fits per location
            model_name = str(service + '_' + group + '_model') #defines name for saving
            pickle.dump(self.ests[group], open((os.path.join('models', model_name)), 'wb')) #saves model
        return self

    def predict(self, X):
        groups = X.groupby(self.column).groups #splits by location
        output = pd.Series(dtype=np.float32) #placeholder for predictions
        for group, rows in groups.items():
            pred = pd.Series(self.ests[group].predict(X.loc[rows]), index = rows)
            output = output.append(pred)
        return output.loc[X.index]


def season_factory():
    # one hot encodes categorical columns
    features = ColumnTransformer([
            ('days', OneHotEncoder(), ['days_cat']),
            ('hours', OneHotEncoder(), ['hours_cat']),
            ('minutes', OneHotEncoder(), ['minutes_cat'])
            ])
    # and returns a pipe from categorical columns to model
    return Pipeline([('time_to_cat', features),
                 #('rf', RandomForestRegressor(max_depth=30, min_samples_leaf=20, n_estimators=550))])
                 ('ridge', Ridge(alpha=100))])
'''
## # # # # # # # # # # # # # # # # # # # # #
#    HYPERPARAMETERS TUNING EVALUATION    #
# # # # # # # # # # # # # # # # # # # # # #
# Pivots dataframe, selects first 80% of data and melts it again. This 80% will be used
# to evaluate the model, as to predict the future of that slice of data.
#57003 values per location
#45600 =~ 80% of each location
pivot_for_training_testing = service_df_cleaned.pivot(index='time_shift', columns='location')['diffOfTime_vertical']
pivot_for_training = pivot_for_training_testing[:45600].reset_index() #approximately 80% of dataset
pivot_for_testing = pivot_for_training_testing[45600:].reset_index()
# each location now contains only the first 80% or last 20% of the data
training = pd.melt(pivot_for_training, id_vars=['time_shift'], value_vars=raw_service.columns[1:],
    value_name='diffOfTime_vertical', var_name = 'location')
testing = pd.melt(pivot_for_testing, id_vars=['time_shift'], value_vars=raw_service.columns[1:],
    value_name='diffOfTime_vertical', var_name = 'location')
# re-calculates the categorical columns for the model for both 80% and 20% datasets
training['days_cat'] = training.apply(lambda x: x.time_shift.weekday(), axis=1)
training['hours_cat'] = training.apply(lambda x: x.time_shift.hour, axis=1)
training['minutes_cat'] = training.apply(lambda x: x.time_shift.minute, axis=1)
testing['days_cat'] = testing.apply(lambda x: x.time_shift.weekday(), axis=1)
testing['hours_cat'] = testing.apply(lambda x: x.time_shift.hour, axis=1)
testing['minutes_cat'] = testing.apply(lambda x: x.time_shift.minute, axis=1)

training_t = training[:45600] # only bakers_basin first 80% for testing
testing_t = testing[:11403] # only bakers_basin last 20% for testing for testing
testing = service_df_cleaned[:57003] # only bakers_basin complete


# features manually calculated for using GridSearchCV
manual_features = ColumnTransformer([
            ('days', OneHotEncoder(), ['days_cat']),
            ('hours', OneHotEncoder(), ['hours_cat']),
            ('minutes', OneHotEncoder(), ['minutes_cat'])
            ])

parameters = {'max_depth':[10, 15, 20], 'min_samples_leaf':[5,15,20], 'n_estimators':[250,350,450]} #best: 20,20,450
parameters = {'max_depth':[20,30], 'min_samples_leaf':[20,30], 'n_estimators':[450,550]} #best: 30,20,550
rf = RandomForestRegressor()
search = GridSearchCV(rf, parameters, verbose = 4)
search.fit(manual_features.fit_transform(training_t), training_t['diffOfTime_vertical']/np.timedelta64(1, 's'))
print(search.best_estimator_)

parameters = {'alpha':[0.001,0.01,0.1,1,10,100,1000,10000]} #best:100
parameters = {'alpha':[75,100,125]} #best:100
rg = Ridge()
search = GridSearchCV(rg, parameters, verbose = 4)
search.fit(manual_features.fit_transform(training_t), training_t['diffOfTime_vertical']/np.timedelta64(1, 's'))
print(search.best_estimator_)
'''

# # # # # # # # # # # # # # # # # # #
#             MODEL USAGE           #
# # # # # # # # # # # # # # # # # # #
model = cat_estimator('location', season_factory).fit(service_df_cleaned, service_df_cleaned['diffOfTime_vertical'])
plt.plot(service_df_cleaned.time_shift, service_df_cleaned.diffOfTime_vertical, service_df_cleaned.time_shift, model.predict(service_df_cleaned))

# # # # # # # # # # # # # # # # # # #
#          MODEL EVALUATION         #
# # # # # # # # # # # # # # # # # # #
# a model for a city at a time, that has X as training and y the same training in hours.
# The mean squared error MSE is calculated comparing the real with its predicted values.
# A smaller error is preferred.

model_evaluation = cat_estimator('location', season_factory).fit(training_t, training_t['diffOfTime_vertical']/np.timedelta64(1, 'h'))
MSE = mean_squared_error(testing_t['diffOfTime_vertical']/np.timedelta64(1, 'h'), model_evaluation.predict(testing_t))
RMSE = math.sqrt(MSE)
print('MSE = ' + str(MSE)) 
print('RMSE = ' + str(RMSE))

#max_depth=15, min_samples_leaf=15, n_estimators=350 -> 4.42
#max_depth=30, min_samples_leaf=20, n_estimators=550 -> 4.41 (best by GridSearchCV)
#ridge(alpha=1) -> 4.09
#ridge(alpha=100) -> 4.09 (best by GridSearchCV)