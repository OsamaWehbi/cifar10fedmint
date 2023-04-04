import pickle
import random
from src.data.data_container import DataContainer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.metrics import mean_squared_error as MSE
from sklearn import model_selection, linear_model
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
import numpy as np
import os.path
import MLM
# Accmodel, Ere, Ede = MLM.bootModel()
# region = "Europe"
# dev = "Phone"
# region = Ere.transform([region])
# dev = Ede.transform([dev])
# arr = np.reshape([region, dev],(1,2))
# print(arr)
# print(Accmodel.predict(arr))
# "C:\Users\user\Desktop\newdata\Original\KDDPlus.csv"
df = pd.read_csv("C:/Users/user/Desktop/LastModel/NTDATA.csv")
# print(df.head())
# print(list(df.keys()))
#
resarr=[]
#
t1 = LabelEncoder()
t2 = LabelEncoder()
t3 = LabelEncoder()
# res = LabelEncoder()
df['Producer'] = t1.fit_transform(df['Producer'])
df['Region'] = t2.fit_transform(df['Region'])
df['DType'] = t3.fit_transform(df['DType'])
training = df.drop('Accuracy', axis='columns')
target = df['Accuracy']
model = tree.DecisionTreeRegressor()
model = model.fit(training.values, target)
scores = cross_val_score(model, training.values, target, scoring='r2', cv=10)
print(scores)
m = np.mean(scores)
resarr.append(m)
print(m)
with open("producer_en.pickle", "wb") as file:
    pickle.dump(t1, file)
with open("region_en.pickle", "wb") as file:
    pickle.dump(t2, file)
with open("devt_en.pickle", "wb") as file:
    pickle.dump(t3, file)
with open("boot_model.pickle", "wb") as file:
    pickle.dump(model, file)

# df['class'] = res.fit_transform(df['class'])
# print(df[' flag'].dtype)
# df[' flag'] = pd.to_numeric(df[' flag'], downcast='float')
# print(df.head())

# print(df[' flag'].dtype)
# x = df.drop('class', axis='columns')
# y = pd.to_numeric(df['class'], downcast='float')
# from collections import Counter
# res = Counter(y.values.tolist())
# print(res)
# quit()
# obj = DataContainer((np.array(x.values, dtype=float)).tolist(), y.values.tolist())
# with open("KDDPTest-21.pkl", "wb") as file:
#     pickle.dump(obj, file)

# print(x.head())
# print(y.head())
# print(y.values)
# training['region'] = bRegion.fit_transform(training['region'])
# training['devt'] = bdevtype.fit_transform(training['devt'])
# with open("region_en.pickle", "wb") as file:
#     pickle.dump(bRegion, file)
# with open("devt_en.pickle", "wb") as file:
#     pickle.dump(bdevtype, file)
# while True:
# x_r, x_t, y_r, y_t = model_selection.train_test_split(training, target, test_size=0.3)
# print(x_r.shape)
# print(training)

# print(target)
#     model = tree.DecisionTreeRegressor(max_depth=14, min_samples_leaf=1, random_state=3)
# model = tree.DecisionTreeRegressor()
# fits = tree.DecisionTreeRegressor()
# for _ in range(3):
#     fits = model.fit(x_r, y_r)
# print(fits)
#
# model = model.fit(x_r.values, y_r)
# score = model.score(x_t.values, y_t)
# print(score)

# scores, models = cross_val_score(model, training, target, scoring='r2', cv=10)
# tet = cross_validate(model, training, target, scoring='r2', cv=10, return_estimator=True)
# print(tet.keys())
# print(np.mean(tet['test_score']))
# print(tet['estimator'])

# print(np.mean(scores))
# prds = cross_val_predict(model, x_t, y_t)
# print(prds)
# pscores = cross_val_score(fits, x_t, y_t, cv=10)
# print(np.mean(pscores))

# print(fits.tree_.max_depth,"\t",fits.min_samples_leaf)
# score = model.score(x_t, y_t)
# print(score)
    # if score >= 0.95:
    #     break

# print(model.score(testing, testarget))
# res = model.predict(x_t)
# mse_dt = MSE(y_t, res)
# rmse_dt = mse_dt**(1/2)
# print(list(res))
# print(mse_dt)
# print(rmse_dt)
# with open("boot_model.pickle", "wb") as file:
#     pickle.dump(model, file)