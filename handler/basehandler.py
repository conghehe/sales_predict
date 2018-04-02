import pandas as pd
from data_processing.missing_data_process import Missing_Data_Process
from data_processing import base_process
from sklearn.model_selection import train_test_split
from machine_learning_algorithms import logistic_regression
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.metrics import mean_squared_error
import math


def test1():
    df = pd.read_csv('D:/360极速浏览器下载/bigmart.csv')
    base_process.label_transform(df, ['Outlet_Size', 'Item_Identifier',
                                      'Item_Fat_Content', 'Item_Type',
                                      'Outlet_Identifier', 'Outlet_Location_Type',
                                      'Outlet_Type'])
    Missing_Data_Process.fill_mean(df, ['Item_Weight'])
    df.to_excel('E:/for_test/a.xlsx')
    # print(df)
    # print(np.isnan(df))
    df['Outlet_Size'] = Missing_Data_Process.classify_missing_process(df, list(df['Outlet_Size'].values),
                                                                      ['Outlet_Size'])
    print(df)
    df.to_excel('E:/for_test/bigmart.xlsx', index=None)


def test2():
    df = pd.read_excel('E:/for_test/bigmart.xlsx')
    targets = df.pop('Outlet_Size')
    # print(df)
    x_train, x_test, y_train, y_test = train_test_split(df, targets)
    lr = logistic_regression.logistic_regression_process(x_train, y_train)
    print(x_train)
    test_values = [lr.predict(np.array(x_test.ix[i]).reshape(1, -1)) for i in x_test.index]
    print(accuracy_score(y_test, test_values))


def test3():
    clf = RandomForestRegressor()
    df = pd.read_excel('E:/for_test/bigmart.xlsx')
    targets = np.array(df.pop('Item_Outlet_Sales')).astype('int')
    print(targets)
    x_train, x_test, y_train, y_test = train_test_split(df, targets)
    lr = clf.fit(x_train, y_train)
    print(len(x_test))
    test_values = []
    j = 0
    for i in x_test.index:
        predict = lr.predict(np.array(x_test.ix[i]).reshape(1, -1))
        test_values.append(predict[0])
        j += 1
        print(j)
        print(predict)
    # test_values = [lr.predict(np.array(x_test.ix[i]).reshape(1, -1)) for i in x_test.index]
    # print(test_values)
    df2 = pd.DataFrame({'predict': test_values, 'real_values': y_test})
    # print(df2)
    df2.to_excel('E:/for_test/bigmart4.xlsx')
    # print(cross_val_score(clf, x_train, y_train, scoring='neg_mean_squared_error'))
    print(math.sqrt(mean_squared_error(y_test, test_values)))


test3()
