import pandas as pd
import numpy as np
import streamlit as st
import joblib


df=pd.read_csc(Salary_dataset.csv')

df.head()
# removing feature

df_remove=df.drop(['Unnamed: 0'],axis=1,inplace=True)

df.isnull().sum() # no missing value
df.duplicated().sum() # no duplicates

x=df['YearsExperience'] 
y=df['Salary']

# scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

# fit the scaler
x=np.array(x).reshape(-1,1)
x_v=scaler.fit_transform(x)

from sklearn . model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x_v,y,test_size=0.2,random_state=1) # splitting the data

from sklearn.linear_model import LinearRegression #model building
reg=LinearRegression(fit_intercept= True, n_jobs=-1, positive =False)
reg.fit(x_train,y_train)
reg.score(x_train,y_train)

ypred=reg.predict(x_test)
joblib.dump(reg,'salary_prediction_model.joblib')# model saving

# app building

def main():
    st.header('salary_prediction'.upper())
    st.info('welcome to the power of science!')
    YearsExperience=st.select_slider('select year of experience',options=[0,1,1.2,  1.4,  1.6, 2, 2.1,  2.3,  3. ,  3.1,  3.3,  3.8,  4. ,  4.1,
        4.2,  4.6,  5. ,  5.2,  5.4,  6. ,  6.1,  6.9, 7, 7.2,  8.0 ,  8.3, 8.8,  9.1,  9.6,  9.7, 10,10.4, 10.6,11,12,13,14,15,16,17,18,19,20,
        21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,55,60,65,70,100])
    st.write('yearly salary in $')
   
    if st.button('predict salary'):
        predictions=reg.predict(scaler.transform([[YearsExperience]]))

        st.success(predictions)

if __name__=='__main__':
    main()

    
   
