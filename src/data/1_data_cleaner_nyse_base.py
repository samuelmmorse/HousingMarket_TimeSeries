# This script takes incomplete data and fills in the gaps... highly unethical

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dir_name = os.path.dirname(__file__)
relative_path = 'raw/nyse_data.csv'
file_path = os.path.join(dir_name, relative_path)


all_data = []
file_array = []

for date in (pd.date_range(start= '1/1/1967', end = '12/31/2020', freq = 'D') ):
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

for i in range(len(all_data)-1, -1, -1):
    if len(all_data[i]) == 1:
        all_data[i].append(all_data[i+1][1])

relative_path = 'data_plotter.csv'
file_path = os.path.join(dir_name, relative_path)
new_file = open(file_path, 'w')

for line in all_data:
    new_file.write(str(line[0].date()) + ',' + line[1] + '\n')
new_file.close()


'''
#PLOT DATA
all_data = np.array(all_data)

x, y = all_data.T
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.plot(x, y)
plt.title("Line graph")
plt.plot(color="red")

plt.show()
'''

