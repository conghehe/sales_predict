
import pandas as pd
from data_processing.missing_data_process import Missing_Data_Process
from data_processing import base_process
import numpy as np


df = pd.read_csv('D:/360极速浏览器下载/bigmart.csv')
base_process.label_transform(df, ['Outlet_Size', 'Item_Identifier',
                                  'Item_Fat_Content', 'Item_Type',
                                  'Outlet_Identifier', 'Outlet_Location_Type',
                                  'Outlet_Type'])
Missing_Data_Process.fill_mean(df, ['Item_Weight'])
df.to_excel('E:/for_test/a.xlsx')
# print(df)
# print(np.isnan(df))
df['Outlet_Size'] = Missing_Data_Process.classify_missing_process(df, list(df['Outlet_Size'].values), ['Outlet_Size'])
print(df['Outlet_Size'])
