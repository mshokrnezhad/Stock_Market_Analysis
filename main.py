from functions import *
import numpy as np
import matplotlib.pyplot as plt
import re

excel_files_path = "excel_files"
stock_list_file_path = "text_files/stock_list.txt"
price_list_file_path = "text_files/price_list.txt"
number_of_data_sets = 68 # we have 68 columns of data.
number_of_training_sets = 40 # we decide to use 40 sets of data to train the h function.
number_of_training_features = 3 # 1 + number of days we want to use their price as the training features

excel_files_list = [f for f in listdir(excel_files_path) if isfile(join(excel_files_path, f))]
# stock_list = get_stock_list_from_excel_files(excel_files_path, stock_list_file_path, excel_files_list)
stock_list = get_stock_list_from_processed_file(stock_list_file_path)
# price_list = get_price_list_from_excel_files(stock_list, excel_files_path, price_list_file_path)
price_list = get_price_list_from_processed_file(price_list_file_path, len(stock_list), number_of_data_sets)

(normal_X, normal_Y, mean, std) = build_linear_data_set(0, price_list, number_of_training_sets, number_of_training_features)
theta = np.matmul(np.matmul(np.linalg.pinv(np.matmul(normal_X.transpose(), normal_X)), normal_X.transpose()), normal_Y)






X, real_Y, X_next_day = build_real_linear_data_set(0, price_list, number_of_data_sets, number_of_training_features)

real_X_next_day = [0 for col in range(number_of_training_features)]
real_X = np.zeros(shape=(number_of_data_sets, number_of_training_features))
temp_X = np.array(X)
real_X[:, 0] = temp_X[:, 0]
real_X_next_day[0] = 1
for col in range(1, number_of_training_features):
    real_X[:, col] = (temp_X[:, col] - mean[col]) / std[col]
    real_X_next_day[col] = (X_next_day[col] - mean[col]) / std[col]


predicted_Y = np.matmul(real_X, theta)

print(np.matmul(real_X_next_day, theta))



label = np.arange(0, number_of_data_sets, 1)

normal_excel_files_list = [""] * len(label)
for fn_index in range(0, len(label)):
    normal_excel_files_list[fn_index] = excel_files_list[label[fn_index]]
for fn in range(0, len(normal_excel_files_list)):
        normal_excel_files_list[fn] = re.sub('MarketWatchPlus-1399_', "", normal_excel_files_list[fn])
        normal_excel_files_list[fn] = re.sub('_', "", normal_excel_files_list[fn])
        normal_excel_files_list[fn] = re.sub('.xlsx', "", normal_excel_files_list[fn])

plt.plot(label, real_Y, 'r--', label, predicted_Y, 'b*')
plt.xticks(label, normal_excel_files_list, rotation='vertical')
plt.grid()
plt.show()
