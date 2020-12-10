# -*- coding: utf-8 -*-
"""
FINAL EXAM SCORE PREDICTOR
Created on Tue Dec  8 07:06:45 2020

@author: ProfN
@goal: Predict a student's final exam score given all of their other scores
    throughout the semester.
"""
import pandas as pd
import numpy as np



##LOAD DATA
MTH11003 = pd.read_csv(r'C:\Users\ProfN\Downloads\StatsGraphs\MTH110-03.csv')
MTH11006 = pd.read_csv(r'C:\Users\ProfN\Downloads\StatsGraphs\MTH110-06.csv')
MTH11007 = pd.read_csv(r'C:\Users\ProfN\Downloads\StatsGraphs\MTH110-07.csv')

##CLEAN UP THE COLUMN NAMES
col_names = [x.split(' (')[0].split(' V')[0] for x in MTH11003.columns]
MTH11003.columns = col_names
deletable_cols = col_names[-24:] ##these columns aggregate data
##keep aggregated R/V score and delete those individual assignment columns
##hopefully this will help combat overfitting
deletable_cols = col_names[6:50]+deletable_cols[:6] + deletable_cols[7:]
col_names = [x.split(' (')[0].split(' V')[0] for x in MTH11006.columns]
MTH11006.columns = col_names
col_names = [x.split(' (')[0].split(' V')[0] for x in MTH11007.columns]
MTH11007.columns = col_names

##MERGE THE DATA INTO A SINGLE DATAFRAME
MTH110df = pd.concat([MTH11003, MTH11006, MTH11007], ignore_index=True)

##DELETE Test Students and Students who did not take the final exam
MTH110df = MTH110df[MTH110df.Student!='Student, Test']
MTH110df = MTH110df[MTH110df.Student!='    Points Possible']
MTH110df.fillna(0, inplace=True)
MTH110df = MTH110df[MTH110df['Final Exam']>43] 
##>3 because one student filled in only one question on the final exam
##and one other student quit halfway through
##replace '(read only)' with 100 in the R/V aggregate column
MTH110df['Reading/Videos Final Score'] = MTH110df[
    'Reading/Videos Final Score'].apply(lambda x: 100 if x=='(read only)'
                                             else x)

##DELETE UNNEEDED COLUMNS (IDs, Section, etc.)
##This includes overall scores as those could be used to intuit the final exam
##And we'll also drop aggregate columns and duplicate columns
MTH110df.drop(columns=['Student','ID','SIS User ID','SIS Login ID','Section',
                       'Midterm Evaluation Survey']+deletable_cols
              , inplace=True)

##TRY USING STANDARD LINEAR REGRESSION##
##FIRST WITH ALL OF THE DATA##
from sklearn.linear_model import LinearRegression
LRmodel = LinearRegression()
x = MTH110df.drop(columns=['Final Exam'])
y = MTH110df['Final Exam']
LRmodel.fit(x,y)
print('Score:', LRmodel.score(x,y))
Yhat = LRmodel.predict(x)
M_A_E = sum(abs(y-Yhat))/len(Yhat)
print('Linear Regression Model Mean Absolute Error:', M_A_E)

##NOW REMOVE SOME OF THE DATA FOR VALIDATION##
##Linear Regression is severely overfitting because there is way more data
##than there are students.
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=33)

##Try Linear Regression with train/test split
print('Attempting a multi-variate LR model')
LRmodel = LinearRegression()
LRmodel.fit(X_train,y_train)
print('Score:', LRmodel.score(X_test,y_test))
LR_pred = LRmodel.predict(X_test)
print('Linear Regression Model Mean Absolute Error:', MAE(y_test,LR_pred))


print('Attempting a RF Regression model')
from sklearn.ensemble import RandomForestRegressor
RFmodel = RandomForestRegressor()
RFmodel.fit(X_train,y_train)
y_pred = RFmodel.predict(X_test)
print('Mean Squared Error:', MSE(y_test,y_pred)) ##baseline model error

def findBest(X_train,y_train,X_test,y_test):
    bestMSE = 400
    best_d = 8
    best_n = 50
    for d in [5,6,7,8,9,10,11]:
        for n in [50, 100, 220, 230, 240, 250, 260, 300, 400]:
            RFmodel = RandomForestRegressor(max_depth=d, n_estimators=n, random_state=21)
            RFmodel.fit(X_train,y_train)
            y_pred = RFmodel.predict(X_test)
            currMSE = MSE(y_test,y_pred)
            if currMSE<bestMSE:
                bestMSE = currMSE
                best_d = d
                best_n = n
            #print(d, n, currMSE)
    print('Best MSE:', bestMSE, ' with depth:', best_d, ' and n_estimators:', best_n)
    RFmodel = RandomForestRegressor(max_depth = best_d, n_estimators = best_n)
    RFmodel.fit(X_train,y_train)
    y_pred = RFmodel.predict(X_test)
    print('Mean Absolute Error:', MAE(y_test,y_pred))
    print('Score:', RFmodel.score(X_test, y_test))
    return best_d, best_n

d, n = findBest(X_train, y_train, X_test, y_test)

RFmodel = RandomForestRegressor(max_depth=d, n_estimators=n)
RFmodel.fit(X_train, y_train)
y_pred = RFmodel.predict(X_test)
##PRINT OUT RESIDUALS##
#for i in range(len(y_test)):
#    print(y_test.iloc[i]-y_pred[i])
    

##TRY XGBOOST ALGORITHM##
#from xgboost import xgb
#XGmodel = xgb()
#XGmodel.fit(X_train, y_train)

##TRY SCALING ALL ASSIGNMENTS TO PERCENTAGES##
#df_scaled = MTH110df.copy()
#Out_of = df_scaled.iloc[0].to_list()
#Out_of = Out_of[:-3] ##remove the two extra credit assignments and the final
#col = df_scaled.columns
#for i in range(len(Out_of)):
#    df_scaled[col[i]] = df_scaled[col[i]]/Out_of[i]
##REMOVE THE ROW WITH THE MAX VALUES
#df_scaled = df_scaled[df_scaled['Test 1']!=1]
#x_scaled = df_scaled.drop(columns=['Final Exam'])
#y_scaled = df_scaled['Final Exam']

#XS_train, XS_test, yS_train, yS_test = train_test_split(x_scaled, y_scaled, 
#                                                    test_size=0.2, 
#                                                    random_state=21)

#d, n = findBest(XS_train, yS_train, XS_test, yS_test)


#MTH11003 = MTH11003[MTH11003.Student!='    Points Possible']
