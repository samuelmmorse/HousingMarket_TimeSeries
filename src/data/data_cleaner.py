# This script takes incomplete data and fills in the gaps... highly unethical

import os
import matplotlib
import numpy as np
import pandas as pd

dir_name = os.path.dirname(__file__)
relative_path = 'raw/nyse_data.csv'
file_path = os.path.join(dir_name, relative_path)


all_data = []
file_array = []

for date in (pd.date_range(start= '1/1/1966', end = '12/31/2020', freq = 'D') ):
    all_data.append([date])


with open(file_path, encoding="utf-8-sig") as bigdata:
    for line in bigdata:
        data_point = line.strip('\n').split(',')
        data_point[0] = pd.to_datetime(data_point[0])
        #data_point[1] = float(data_point[1])

        file_array.append(data_point)

for line in file_array:
    for datum in all_data:
        if (line[0] == datum[0]):
            datum.append(line[1])

for line in all_data:
    print(line)

