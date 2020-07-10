from functions import *
import numpy as np

excel_files_path = "excel_files"
stock_list_file_path = "text_files/stock_list.txt"
price_list_file_path = "text_files/price_list.txt"

# stock_list = get_stock_list_from_excel_files(excel_files_path, stock_list_file_path)
stock_list = get_stock_list_from_processed_file(stock_list_file_path)
# price_list = get_price_list_from_excel_files(stock_list, excel_files_path, price_list_file_path)
price_list = get_price_list_from_processed_file(price_list_file_path, len(stock_list), 68)

# we have 68 columns of data.
# we decide to use 40 sets of data to train the h function.

number_of_training_sets = 40
number_of_training_features = 3 # 1 + number of days we want to use their price as the training features
theta = [0 for col in range(0, number_of_training_features)]
error = 0

(Normal_X, Y) = build_linear_data_set(1, price_list, number_of_training_sets, number_of_training_features)
theta = np.matmul(np.matmul(np.linalg.pinv(np.matmul(Normal_X.transpose(), Normal_X)), Normal_X.transpose()), Y)
error = (1/(2*number_of_training_sets))*(np.dot((np.matmul(Normal_X, theta)-Y), (np.matmul(Normal_X, theta)-Y)))

print(str(error))

