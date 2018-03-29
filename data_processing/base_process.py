
from sklearn import preprocessing
from data_processing.missing_data_process import Missing_Data_Process
from pandas import Series
import numpy as np


def label_transform(df, columns):
    le = preprocessing.LabelEncoder()
    for column in columns:
        na_indexs = Missing_Data_Process.get_na_index(df[column])
        # print(na_indexs)
        df_copy = df.dropna(subset=[column])
        transform = list(le.fit_transform(df_copy[column]))
        for index in na_indexs:
            transform.insert(index, np.NaN)
        # print(transform)
        df[column] = Series(transform)
    # print(df1)
