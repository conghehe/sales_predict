
import pandas as pd
import numpy as np
import sklearn
from machine_learning_algorithms import logistic_regression


class Missing_Data_Process(object):
    def __init__(self):
        pass

    # 传入List列表并直接修改原列表的缺失值为平均值
    @staticmethod
    def fill_mean(df, columns):
        for column in columns:
            mean_num = round(np.mean(df[column]), 2)
            for i in range(len(df)):
                # print(df.loc[i ,column])
                if np.isnan(df.loc[i, column]):
                    df.loc[i, column] = mean_num

    # 传入List列表并直接修改原列表的缺失值为加权平均值
    @staticmethod
    def fill_average(datas, weights):
        average_num = round(np.average(datas, weights=weights), 2)
        for i in range(len(datas)):
            if not datas[i]:
                datas[i] = average_num

    @staticmethod
    def del_null_data(df, columns):
        df.dropna(subset=columns, inplace=True)
        return df

    @staticmethod
    def get_na_index(datas):
        indexs = [index for index in datas.isna().index if datas.isna()[index]]
        # for i in range(len(datas)):
        #     print(datas[i])
        #     if np.isnan(datas.ix[i]):
        #         indexs.append(i)
        return indexs

    @staticmethod
    def classify_missing_process(df, target, columns):
        df_dropna = df.dropna(subset=columns)
        # df_dropna.to_excel('E:/for_test/b.xlsx')
        df_dropna = df_dropna[[feature for feature in df_dropna.columns if feature not in columns]]
        df_notna = df.isna()
        if len(columns) == 1:
            na_boolen = df_notna[columns[0]]
        else:
            na_boolen = df_notna[columns[0]] & df_notna[columns[1]]
            for i in range(2, len(columns)):
                na_boolen = na_boolen & df_notna[columns[i]]
        print(na_boolen)
        df_na = df[na_boolen][[feature for feature in df_dropna.columns]]
        temp_target = pd.Series(target)
        target_notna = temp_target[pd.Series(target).notna()]
        print(df_na.index)
        print(len(df_na.index))
        lr = logistic_regression.logistic_regression_process(np.array(df_dropna), np.array(target_notna))
        for i in df_na.index:
            # print(df_na.ix[i])
            value = lr.predict(np.array(df_na.ix[i]).reshape(1, -1))
            # print(value)
            target[i] = value[0]
        return target



# data = {'age': [24, 45, 12, 26], 'name': ['mark', np.nan, 'jack', 'jorn'], 'gender': ['F', 'M', None, 'M']}
# df = pd.DataFrame(data)
# indexs = Missing_Data_Process.get_na_index(df['name'])

# print(indexs)
