import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import numpy as np
import os.path
resarr = {}
cntrv = 0
def CPUMODEL():
    if os.path.exists('CPUS_model.pickle'):
        print("loading Trained Model")
        model = pickle.load(open("CPUS_model.pickle", "rb"))
    else:
        df = pd.read_csv("C:/Users/user/Desktop/new.csv")
        X = np.array(df['datasize']).reshape(-1, 1)
        Y = df['CPUS']
        model = linear_model.LinearRegression()
        model.fit(X, Y)
        with open("ENUS_model.pickle", "wb") as file:
            pickle.dump(model, file)
    return model

def RAMODEL():
    if os.path.exists('RAUS_model.pickle'):
        print("loading Trained Model")
        model = pickle.load(open("RAUS_model.pickle", "rb"))
    else:
        df = pd.read_csv("C:/Users/user/Desktop/new.csv")
        X = np.array(df['datasize']).reshape(-1, 1)
        Y = df['RAUS']
        model = linear_model.LinearRegression()
        model.fit(X, Y)
        with open("RAUS_model.pickle", "wb") as file:
            pickle.dump(model, file)
    return model

def TIMEMODEL():
    if os.path.exists('Time_model.pickle'):
        print("loading Trained Model")
        model = pickle.load(open("Time_model.pickle", "rb"))
    else:
        df = pd.read_csv("C:/Users/user/Desktop/new.csv")
        X = np.array(df['datasize']).reshape(-1, 1)
        Y = df['Time']
        model = linear_model.LinearRegression()
        model.fit(X, Y)
        with open("Time_model.pickle", "wb") as file:
            pickle.dump(model, file)
    return model

def ENMODEL():
    if os.path.exists('ENUS_model.pickle'):
        print("loading Trained Model")
        model = pickle.load(open("ENUS_model.pickle", "rb"))
    else:
        df = pd.read_csv("C:/Users/user/Desktop/new.csv")
        X = np.array(df['datasize']).reshape(-1, 1)
        Y = df['ENUS']
        model = linear_model.LinearRegression()
        model.fit(X, Y)
        with open("ENUS_model.pickle", "wb") as file:
            pickle.dump(model, file)
    return model

def bootModel():
    producer = pickle.load(open("producer_en.pickle", "rb"))
    region = pickle.load(open("region_en.pickle", "rb"))
    devt = pickle.load(open("devt_en.pickle", "rb"))
    if os.path.exists('boot_model.pickle'):
        print("loading Trained Model and encoders")
        model = pickle.load(open("boot_model.pickle", "rb"))

    return model, producer, region, devt

def updatebooModel():
    t1 = pickle.load(open("producer_en.pickle", "rb"))
    t2 = pickle.load(open("region_en.pickle", "rb"))
    t3 = pickle.load(open("devt_en.pickle", "rb"))
    df = pd.read_csv("C:/Users/user/Desktop/localfed/FedMatch/NTDATA.csv")
    df['Producer'] = t1.transform(df['Producer'])
    df['Region'] = t2.transform(df['Region'])
    df['DType'] = t3.transform(df['DType'])
    training = df.drop('Accuracy', axis='columns')
    target = df['Accuracy']
    model = tree.DecisionTreeRegressor()
    model = model.fit(training.values, target)
    scores = cross_val_score(model, training.values, target, scoring='neg_mean_squared_error', cv=10)
    m = -1 * np.mean(scores)
    # resarr.append(m)
    global cntrv
    resarr[cntrv] = m
    cntrv+=1
    # print(m)
    with open("boot_model.pickle", "wb") as file:
        pickle.dump(model, file)
