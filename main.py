#Diabetes Detection using KNN Classifier

#import the lib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle

#load the data
data = pd.read_csv("diabetes.csv")
print(data.head())

#max min values
max_values = data.max(axis="rows")
min_values = data.min(axis="rows")

#undearstand the data
res = data.isnull().sum()
print(res)

data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = data[["Glucose", "BloodPressure", 
"SkinThickness", "Insulin", "BMI"]].replace(0, np.NAN)
# we are doing this because bmi glucose bp etc cant be zero so we are doing feature engineering over here

#understand the data
res = data.isnull().sum()
print(res)

#handle the null data
ndata = data.fillna({
		"Glucose" : data["Glucose"].mean(),
		"BloodPressure" : data["BloodPressure"].mean(),
		"SkinThickness" : data["SkinThickness"].mean(),
		"Insulin" : data["Insulin"].mean(),
		"BMI" : data["BMI"].mean(),	
		})

#understand the data
res = ndata.isnull().sum()
print(res)

#features and target
features = ndata.drop("Outcome", axis="columns")
target = data["Outcome"]

#Feature Scaling
mms = MinMaxScaler()
nfeatures = mms.fit_transform(features)

print(features)
print(nfeatures)

#find the value of N
N = len(data)
N = round(N**0.5,2)
N = int(N)
if N % 2 == 0:
	N = N + 1
print(N)

#model and predict
model = KNeighborsClassifier(n_neighbors=N, metric="euclidean")
model.fit(nfeatures, target)

#predict
data = [[6, 148, 72 ,35, 0, 33.6, 0.627, 50]]
data = [[1, 89, 66, 23, 94, 28.1, 0.167, 21]]
ndata = mms.transform(data)
res = model.predict(ndata)
print(res)

print(max_values)
print(min_values)

#save the model 
with open("diabetes.model", "wb") as f:
	pickle.dump(model,f)
