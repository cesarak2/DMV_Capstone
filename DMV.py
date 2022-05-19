import numpy as np
import pandas as pd
from cgi import print_environ_usage
from os.path import exists # check if SQL exists
import re
import smtplib, ssl
from subprocess import CalledProcessError
import time
from datetime import datetime
import random #for random refresh times
from pprint import pprint
import sqlite3
import sqlite3 as db
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from sqlalchemy import outerjoin

# TESTING QUERIES
# to execute
#con = sqlite3.connect('DB_DMV.sqlite')
connection = sqlite3.connect('DB_DMV.sqlite')
#db = mysql.connector.connect(option_files='DB_DMV.sqlite', use_pure=True, use_unicode=True, charset='ascii')


# 3 WAYS TO EXECUTE:
    # 1st:
query = """
    SELECT * FROM INITIAL_PERMIT;
    """
query = """
    SELECT * FROM REAL_ID
    """
c = connection.cursor()
r = c.execute(query)
a = r.fetchall()

df = pd.read_sql(query, connection)
# con.commit()
# cursor = con.cursor()
a = connection.execute("SELECT * FROM INITIAL_PERMIT limit 5")
a = connection.execute(query)
a = connection.execute("SELECT * FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%';")

record=a.fetchone()
if record==None:
    pass
print (record)

    #2nd
# to print
query="SELECT COUNT(*) from INITIAL_PERMIT;"
query="PRAGMA table_info(INITIAL_PERMIT);"
query = """
SELECT SUM(CASE when Bayonne is NULL then 1 else 0 end) count_nulls 
, count(Bakers_Basin) count_not_nulls from INITIAL_PERMIT
"""

cur=connection.cursor()
cur.execute(query)
cur.fetchall()
print("description: ", end="")
pprint(cur.description)

while True:
   record=cur.fetchone()
   if record==None:
       break
   print (record)

type(record)
print(record)
record



# # # TO MODIFY TABLE

query="""
UPDATE
    INITIAL_PERMIT
SET
    Bakers_Basin = null WHERE Bakers_Basin = 'NULL'
"""

cur=db.cursor()
cur.execute(query)
record=cur.fetchone()
print (record)



# GETS ALL TABLES

query = """
SELECT
    *
FROM
    sqlite_schema
WHERE
    type='table' AND
    name NOT LIKE 'sqlite_%';
"""

cur=db.cursor()
cur.execute(query)
while True:
   record=cur.fetchone()
   if record==None:
       break
   print (record)

#https://www.knowledgehut.com/tutorials/python-tutorial/python-database-connectivity

# # # # # # # # # # # # # # # # # # # # # # # # ##
# GETS ALL SERVICES BY EXTRATING TABLES FROM SQL #
# # # # # # # # # # # # # # # # # # # # # # # # ##

query = "SELECT name FROM sqlite_schema WHERE type='table' ORDER BY name;"
cur=connection.cursor()
cur.execute(query) #executes query
list_of_services = [] #defines empty list to be populated
while True: # loops through all tables and retrieve their names
   record=cur.fetchone()
   if record==None: # breaks when there is no more data to fetch
       break
   list_of_services.append(record[0]) #comes in a tuple with two elements
   #print (record)
list_of_services

# # # # # # # # # # # # # # # # # # # # # # # # # ##
# GETS ALL CITIES BY EXTRATING COLUMNS SQL COLUMNS #
# # # # # # # # # # # # # # # # # # # # # # # # # ##

query = "PRAGMA table_info(REAL_ID);"
cur=connection.cursor()
cur.execute(query) #executes query
list_of_locations = [] #defines empty list to be populated
while True: # loops through all tables and retrieve their column names
   record=cur.fetchone()
   if record==None: # breaks when there is no more data to fetch
       break
   list_of_locations.append(record[1]) #comes in a tuple with four elements
   #print (record)
list_of_locations


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
connection.commit()

# # # # # # # # # # # # # # # # # # # # # # # # # # #
#            COMPARES NULLS AGAINST VALUES          #
# # # # # # # # # # # # # # # # # # # # # # # # # # #
# COUNTS NULLS
query = """
SELECT SUM(CASE when Bakers_Basin is NULL then 1 else 0 end) count_nulls 
, count(Bayonne) count_not_nulls from INITIAL_PERMIT;
"""
# should be 39417, 57003
a = connection.execute(query)
record=a.fetchone()
if record==None:
    pass
print (record)



# # # # # # #
# USING ROW_FACTORY TOIMPORT USING LISTS
# # # # # # #

connection.row_factory = lambda cursor, row: row[0]
c = connection.cursor()
timestamps = c.execute('SELECT time_stamp FROM REAL_ID WHERE Bakers_Basin IS NOT NULL;').fetchall()


connection.row_factory = lambda cursor, row: row[0]
c = connection.cursor()
records = c.execute('SELECT Bakers_Basin FROM REAL_ID WHERE Bakers_Basin IS NOT NULL;').fetchall()

connection.row_factory = lambda cursor, row: row[0,1]
c = connection.cursor()
df = c.execute('SELECT time_stamp FROM REAL_ID WHERE Bakers_Basin IS NOT NULL;').fetchall()
df

# # # # # # 
# CONVERTING TO SERIES AND THEN TO DATETIME
# # # # # #
records_series = pd.Series(records)
records_series = pd.to_datetime(records_series)
timestamps_series = pd.Series(timestamps)
timestamps_series = pd.to_datetime(timestamps_series)

delta = records_series - timestamps_series
records_series.head(3)
timestamps_series.head(3)
delta.head(3)


# https://stackoverflow.com/questions/2854011/get-a-list-of-field-values-from-pythons-sqlite3-not-tuples-representing-rows





# # # # # # # # # # # # # # # # # # 
#         CONVERTING TO DF        #
# # # # # # # # # # # # # # # # # # 
con = sqlite3.connect('DB_DMV.sqlite')

query = """
    SELECT * FROM INITIAL_PERMIT
    """
c = connection.cursor()
r = c.execute(query)
a = r.fetchall()

df = pd.read_sql(query, con)

df_dmv_zipcodes = pd.DataFrame.from_dict(data={
    'Bakers_Basin': '08648',
    'Bayonne': '07002',
    'Camden': '08030',
    'Cherry_Hill': '08002',
    'Cardiff': '08201',
    'Delanco': '08075',
    'East_Orange': '07017',
    'Eatontown': '07702',
    'Edison': '08817',
    'Elizabeth': '07114',
    'Flemington': '08822',
    'Freehold': '07728',
    'Hazlet': '07730',
    'Jersey_City': '07030',
    'Lakewood': '08701',
    'Lodi': '07644',
    'Manahawkin': '08050',
    'Medford': '08055',
    'Newark': '07101',
    'Newton': '07860',
    'North_Bergen': '07047',
    'Oakland': '07436',
    'Paterson': '07501',
    'Rahway': '07065',
    'Randolph': '07845',
    'Rio_Grande': '08242',
    'Runnemede': '08007',
    'Salem': '08070',
    'Somerville': '08876',
    'South_Brunswick': '08512',
    'South_Plainfiled': '07080',
    'Springfield': '07081',
    'Toms_River': '08732', 
    'Trenton_Regional': '08601',
    'Turnersville': '08012',
    'Vineland': '08332',
    'Wallington': '07055',
    'Washington': '07882',
    'Wayne': '07035',
    'West_Deptford': '08051'

}, orient='index', columns=['zip'])
df_dmv_zipcodes.index.rename('dmv', inplace=True)

# # # # # # # # # # # # # # # # # # # # # # # #
#            GET NEXT AVAILABLE TIME          #
# # # # # # # # # # # # # # # # # # # # # # # #

df_dmv_zipcodes.to_sql('dmv_zips', con, if_exists='append')
#c = con.cursor()
#r = c.execute('select dmv, zip from dmv_zips')
#r.fetchall()
'''
def predict(location, time):
    """

    """
    return best_time, best_location
'''

df.shape

df['shift'] = '' #call everyone am and change the ones repeated to pm
cols = df.columns.tolist() #change columns order...
cols = cols[:1] + cols[-1:] + cols[1:-1] # ...to bring shift to second positon
df = df[cols]
df.groupby('shift').count()

# # # # # # # # # # # # # # # # # # # # # # # #
#            FIXING THE MISSING AM PM         #
# # # # # # # # # # # # # # # # # # # # # # # #

# 1) if day is the same and time is smaller than previous, shift = pm
# 2) if day has a pm, the missing ones are am
#    most of dates should have a resolution by now
# 3) if day is the very next to the previous one and hour diff is small = am
# 4) if day is the very previous to the next one and hour diff is small = pm

df['time_stamp'] = pd.to_datetime(df['time_stamp'], format='%d/%m/%Y %H:%M')
df['time_shift'] = df['time_stamp'].shift(1)
# 1) gives first pm of day (time drops to 0 but day stays the same)
#   df['shift'] = ''
#df['shift'] = df.apply(lambda x: 'pm' if x.time_stamp.hour < x.time_shift.hour and x.time_stamp.day == x.time_shift.day else '', axis=1)
df['shift'] = df.apply(lambda x: 'pm' if \
                (x.time_stamp + pd.offsets.Hour(12)).hour < (x.time_shift + pd.offsets.Hour(12)).hour and \
                x.time_stamp.day == x.time_shift.day \
                else 'am', axis=1)

# 1.1) completes pm for all values of the day
for i in range(1, len(df)):
    # if previous is pm and current is the same day than previous, it is also pm
    if df['shift'][i-1] == 'pm' and df['time_stamp'][i].day == df['time_stamp'][i-1].day:
        df['shift'].iloc[i] = 'pm'
df.groupby('shift').count() # to verify
# if pm, adds 12 hours, converting it to pm
#df['time_shift'] = df.apply(lambda x: (x['time_stamp'] + pd.Timedelta(hours=12)) if x['shift'] == 'pm' else x[ 'time_stamp'], axis=1) 
#df['time_shift'] = df.apply(lambda x: x['time_stamp'] if x['shift'] == 'am' and x['time_stamp'].hour == 12 else x[ 'time_stamp'], axis=1) #with AND

#df['time_shift'] = df.apply(lambda x: x['time_stamp'] if (x['shift'] == 'am' and x['time_stamp'].hour < 12) or \
#                    (x['shift'] == 'pm' and x['time_stamp'].hour == 12) else x[ 'time_stamp'], axis=1) #works for 'do nothing'

'''
4 cases:
am < 12 -> do nothing (11:00 -> 11:00)
pm = 12 -> do nothing (noon = 12:15 -> 12:15)
am = 12 -> subtract 12h (midnight fifteen = 12:15 -> 00:15)
pm < 12 -> adds 12 hours (11:00 -> 23:00) 
'''
df['time_shift'] = df.apply(lambda x: x['time_stamp'] if (x['shift'] == 'am' and x['time_stamp'].hour < 12) or \
                    (x['shift'] == 'pm' and x['time_stamp'].hour == 12) else \
                        x['time_stamp'] - pd.Timedelta(hours=12) if x['shift'] == 'am' and x['time_stamp'].hour == 12 else \
                            x['time_stamp'] + pd.Timedelta(hours=12), axis=1)
#df.to_csv('df_with_shift.csv')
df.drop(['time_stamp', 'shift'], axis=1, inplace = True)
df[:50]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  DROPPING THE COLUMNS THAT DON'T HAVE ANY APPOINTMENTS  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
df2 = df.copy()

# identify and remove columns that don't have any appointment
total_entries = df2.shape[0]
list_to_drop = []
columnsdf2 = df2.columns
for column in columnsdf2:
    if df2.loc[:, column].isna().sum() == total_entries: # all entries are NAs
        list_to_drop.append(column)
        df2.drop([column], axis = 1, inplace=True) #can't drop columns this way
    elif df2[column][-1:] is not None: # there are appointments, and the last is valid
        pass
    else: #to avoid leaking appointment from next agency, create a fake avent at the last position
        df2[column][-1:] = df2['time_shift'][-1:]

# data was captured in different time periods, so to avoid a big gap between them, a fake event
# will be created at the end of each time period. Ideally this step is be avoided by using a better
# data collection.
# Assumptions: (1) all locations without any appointment were already removed, (2) if that agency
# has enough appointments, this extra event will not change much, and (3) if that agency has seldom
# events, this approach will reduce the variability of results (by bringing mean down).

list_to_drop = []
columnsdf2 = df2.columns
for i in range(0, len(df2)-1): # for every entry
    if (df2['time_shift'][i+1] - df2['time_shift'][i]).days > 5: # if more than 5 days between time and next
        for column in columnsdf2: # for that time gap, get every location
            if df2[column][i] is None: # last value is not valid
                df2[column][i] = df2['time_shift'][i].strftime('%m/%d/%Y %I:%M %p') # 
                print(str(column) + ' had a placeholder added')
            else:
                print(str(column) + ' had a valid apointment')
        
        print(str(df2['time_shift'][i-1]) + ' - ' + str(df2['time_shift'][i]) + ': bigger than 5 days difference')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  CONVERTS HOR df INTO VERTICAL df (MELT) FOR EASIER ACCESS  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

df2 = pd.melt(df2, id_vars=['time_shift'], value_vars=df2.columns[1:]) # melts original df
#df2.dropna(subset=['value'])
df2['hasAvailability'] = df2['value'].apply(lambda x: 0 if x is None else 1) # 1 if that time is available for that service

df2['value'] = pd.to_datetime(df2['value'], format='%m/%d/%Y %I:%M %p') #directly from the DMV website
df2['time_shift'] = pd.to_datetime(df2['time_shift'], format='%Y-%m-%d %H:%M')

df2['diffOfTime'] = df2['value'] - df2['time_shift'] # horizontal distance between times
df2['diffTimeDays'] = df2['diffOfTime'].apply(lambda x: x.days) # horizontal distance between times in days

df2['countdown_n'] = df2.groupby((df2['hasAvailability'] == 1).cumsum()).cumcount(ascending=False)+1 # n of intervals until next avail. appoint.
df2.loc[df2['hasAvailability'] == 1, 'countdown_n'] = 0

# # # # # # # # # # # # # # # # # # # # # # # # #
#            GET NEXT APPOINTMENT TIME          #
# # # # # # # # # # # # # # # # # # # # # # # # #

df2['temp_next_time'] = 0
for i in range(len(df2)):
    try:
        df2['temp_next_time'][i] = df2.iloc[i + df2['countdown_n'][i], 0]
    except IndexError:
        df2['temp_next_time'][i] = '0'
df2['temp_next_time'] = pd.to_datetime(df2['temp_next_time'], format='%Y-%m-%d %H:%M') #directly from the DMV website
df2['diffOfTime_vertical'] = df2['temp_next_time'] - df2['time_shift'] # horizontal distance between times

#df2['days_cat'] = df.apply(lambda x: x.time_shift.day, axis=1)
#df2['hours_cat'] = df.apply(lambda x: x.time_shift.hour, axis=1)
#df2['minutes_cat'] = df.apply(lambda x: x.time_shift.minute, axis=1)

df2.to_csv('df2.csv')
len(df2)

df2_cleaned = df2.copy()
df2_cleaned.drop(['value', 'hasAvailability', 'diffOfTime', 'diffTimeDays', 'countdown_n', 'days_cat', 'hours_cat', 'minutes_cat'], axis = 1, inplace=True)
df2_cleaned['days_cat'] = df2_cleaned.apply(lambda x: x.time_shift.day, axis=1)
df2_cleaned['hours_cat'] = df2_cleaned.apply(lambda x: x.time_shift.hour, axis=1)
df2_cleaned['minutes_cat'] = df2_cleaned.apply(lambda x: x.time_shift.minute, axis=1)

#df2_cleaned.drop(['days_cat', 'hours_cat', 'minutes_cat'], axis = 1, inplace=True)

#df2_cleaned_temp = df2_cleaned[57000:57005]
#df2_cleaned_temp = df2_cleaned[57000:200000]
#df2_cleaned_temp[57001:79990]
#df2_cleaned_temp.reset_index(inplace = True)
#df2_cleaned_temp.time_shift[57001].day




df2['diffOfTime_vertical'].plot(ylim=[0, .1E15])
# # # # # # # # # # # # # # # # # # # # # # # # #
#          FIND FREQUENCY BY USING FFT          #
# # # # # # # # # # # # # # # # # # # # # # # # #
columnsdf2 = df2.variable.unique()
import scipy.fft
for location in columnsdf2:
    data_solo_location = df2.groupby('variable').get_group(location) #splits dataset per city
    diff_vertical_in_s = np.array(data_solo_location.diffOfTime_vertical/ np.timedelta64(1, 's'))
    fft = scipy.fft.fft((diff_vertical_in_s - diff_vertical_in_s.mean()))
    plt.plot(np.abs(fft))
    plt.title('FFT for ' + str(location))
    plt.show()


'''
df_grpcity = df2[['time_shift', 'hasAvailability']].groupby('time_shift').sum('hasAvailability')

df_grpcity['hasAvailability'] = df_grpcity['hasAvailability'].apply(lambda x: x if x == 0 else 1)
df_grpcity['countdown_n'] = df_grpcity.groupby((df_grpcity['hasAvailability'] == 1).cumsum()).cumcount(ascending=False)+1
df_grpcity.loc[df_grpcity['hasAvailability'] == 1, 'countdown_n'] = 0

dist = df_grpcity.groupby('countdown_n').count()
dist[:50]
'''

# # # # # # # # # # # # # # # # # # # # # # # # # # #
#                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # 
# IMPORTING ML LIBRARIES
# # # # # #
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


# df2.columns = time_stamp', 'variable', 'value', 'hasAvailability', 'diffOfTime', 'diffTimeDays', 'countdown'
columnsdf2 = df2.variable.unique()
for location in columnsdf2[:3]:
    data_solo_location = df2.groupby('variable').get_group(location) #splits dataset per city
    # makes X to be a relative time, stating at zero seconds
    data_solo_location['time_shift'] = data_solo_location['time_shift'] - data_solo_location['time_shift'][0]



data_solo_location = df2.groupby('variable').get_group('Bayonne') #splits dataset per city
data_solo_location['time_shift'] = data_solo_location['time_shift'] - data_solo_location['time_shift'][0]

train, test = train_test_split(data_solo_location[['time_shift', 'diffOfTime_vertical']], test_size=0.2)
#data_solo_location['time_shift'][10].seconds
#y_train = np.ravel(train[['diffOfTime_vertical']])
#y_test = np.ravel(test[['diffOfTime_vertical']])
y_train = np.array(train[['diffOfTime_vertical']]/ np.timedelta64(1, 's'))
y_test = np.array(test[['diffOfTime_vertical']]/ np.timedelta64(1, 's'))
#train = pd.DataFrame(train)
X_train = np.array(train['time_shift'] / np.timedelta64(1, 's')).reshape(-1,1)
X_test = np.array(test[['time_shift']] / np.timedelta64(1, 's')).reshape(-1,1)

#enc = OneHotEncoder(handle_unknown='ignore')
#one = 

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
#pd.Series(a).arange(-1,1)
from sklearn.metrics import mean_squared_error
mean_squared_error

final_predictions = rf.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

pd.to_datetime(([-4.61856449e+15]))
a = pd.to_datetime((['2022/05/02 21:00:00']))
test = a[0].value

records_series[10] - records_series[0]
records_series_relative = records_series.apply(lambda x: x - records_series.iloc[0])

'''
# ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # !
# ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # !
# ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # !
# ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # !
# ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # !
# ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # !
# ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # !
# ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # !
'''
import pickle
from sklearn import base



class cat_estimator(base.BaseEstimator, base.RegressorMixin):
    
    def __init__(self, column, estimator_factory):
        # column is the value to group by; estimator_factory can be
        # called to produce estimators
        self.column = column
        self.estimator_factory = estimator_factory
        
    def fit(self, X, y):
        # Create an estimator and fit it with the portion in each group
        # uses group_by + get_group to split df in cities. Train a each model and store them in a dictionary
        self.cities = []
        for city in X[self.column].unique(): #gets all cities names in a list
            self.cities.append(city)
        self.dict_models = {} # where all models will be stores, key = location
        for i in range(len(self.cities)): #defines city and list_models
            city = self.cities[i] #defines the city name based on position i
            print(city)
            self.dict_models[city] = self.estimator_factory() #adds entry to dictionary
            #splits dataset per city. As the index will be used later, it must be reset
            data_solo_city = X.groupby(self.column).get_group(city).reset_index() 
            #y0 = np.array(train[['diffOfTime_vertical']]/ np.timedelta64(1, 's'))
            y = np.array(y/ np.timedelta64(1, 's'))
            self.dict_models[city].fit(data_solo_city, y)
            print('model trained for ' + city)
        return self

    def predict(self, X):
        answers = []
        X_t = X
        X_datetime_only = X_t.apply(lambda x: (x - X_t.iloc[0]) / pd.offsets.Hour(1))
        for i in range(0,len(X_datetime_only)):
            try:
                self.prediction_model = self.dict_models[X['city'][i]] # gets model for the city for given line
                prediction_values_df = X_datetime_only.iloc[i] # uses only that row to calculate answer
                [prediction] = self.prediction_model.predict(prediction_values_df)
            except KeyError:
                prediction = 0
            answers.append(prediction)
        return answers


class time_to_categorical(base.BaseEstimator, base.TransformerMixin):
#    def __init__(self):
    def fit(self, X, y=None):
        self.features = ColumnTransformer([
            ('days', OneHotEncoder(), X['days_cat']),
            ('hours', OneHotEncoder(), X['hours_cat']),
            ('minutes', OneHotEncoder(), X['minutes_cat'])
            ])
        return self
    
    def transform(self, X):
        self.features.fit_transform(X)
        return self.features

def pipe_cat():
    return Pipeline([('time_to_cat', time_to_categorical()),
                 ('rf', RandomForestRegressor(max_depth=15, min_samples_leaf=15, n_estimators=450))])                     

model = cat_estimator('variable', pipe_cat).fit(df2_cleaned, df2_cleaned['diffOfTime_vertical'])




df2_t = df2[:5]
df2_t['daysdf2'] = df2_t.apply(lambda x: x.time_shift.day, axis=1)
df2_t['hoursdf2'] = df2_t.apply(lambda x: x.time_shift.hour, axis=1)
df2_t['minutesdf2'] = df2_t.apply(lambda x: x.time_shift.minute, axis=1)

enc = OneHotEncoder(handle_unknown='ignore')
features = ColumnTransformer([
            ('days', OneHotEncoder(), ['daysdf2']),
            ('hours', OneHotEncoder(), ['hoursdf2']),
            ('minutes', OneHotEncoder(), ['minutesdf2'])
            ])
features.fit_transform(df2_t)







data_solo_location = df2.groupby('variable').get_group('Bayonne') #splits dataset per city
data_solo_location['time_shift'] = data_solo_location['time_shift'] - data_solo_location['time_shift'][0]

train, test = train_test_split(data_solo_location[['time_shift', 'diffOfTime_vertical']], test_size=0.2)
#data_solo_location['time_shift'][10].seconds
#y_train = np.ravel(train[['diffOfTime_vertical']])
#y_test = np.ravel(test[['diffOfTime_vertical']])
y_train = np.array(train[['diffOfTime_vertical']]/ np.timedelta64(1, 's'))
y_test = np.array(test[['diffOfTime_vertical']]/ np.timedelta64(1, 's'))
#train = pd.DataFrame(train)
X_train = np.array(train['time_shift'] / np.timedelta64(1, 's')).reshape(-1,1)
X_test = np.array(test[['time_shift']] / np.timedelta64(1, 's')).reshape(-1,1)

#enc = OneHotEncoder(handle_unknown='ignore')
#one = 

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
#pd.Series(a).arange(-1,1)
from sklearn.metrics import mean_squared_error
mean_squared_error

final_predictions = rf.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

pd.to_datetime(([-4.61856449e+15]))
a = pd.to_datetime((['2022/05/02 21:00:00']))
test = a[0].value

records_series[10] - records_series[0]
records_series_relative = records_series.apply(lambda x: x - records_series.iloc[0])
