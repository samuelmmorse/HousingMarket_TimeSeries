# This script takes incomplete data and fills in the gaps... highly unethical

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dir_name = os.path.dirname(__file__)
relative_path = 'raw/GDP_percapita.csv'
file_path = os.path.join(dir_name, relative_path)


all_data = []
file_array = []

for date in (pd.date_range(start= '1/1/1967', end = '12/31/2020', freq = 'D') ):
    all_data.append([date])


with open(file_path, encoding="utf-8-sig") as bigdata:
    for line in bigdata:
        data_point = line.strip('\n').split(',')
        data_point[0] = pd.to_datetime(data_point[0])
        file_array.append(data_point)

for line in file_array:
    for datum in all_data:
        if (line[0] == datum[0]):
            datum.append(float(line[1]))


for i in range(len(all_data)):
    if len(all_data[i]) == 1:
        all_data[i].append(all_data[i-1][1])



relative_path = 'data_plotter5.csv'
file_path = os.path.join(dir_name, relative_path)
new_file = open(file_path, 'w')
old_file = open(os.path.join(dir_name, 'data_plotter4.csv'))

for line, existing in zip(all_data, old_file):
    new_file.write(existing.strip('\n') + ',' + str(line[1]) + '\n')
new_file.close()



