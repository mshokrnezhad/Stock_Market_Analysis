from functions import *
import numpy as np
from tqdm import tqdm
import time

excel_files_path = "excel_files"
stock_list_file_path = "text_files/stock_list.txt"
my_stocks_file_path = "text_files/my_stocks.txt"
price_list_file_path = "text_files/price_list.txt"
number_of_data_sets = 311 # we have 68 columns of data.
number_of_training_sets = 200 # we decide to use 40 sets of data to train the h function.
number_of_training_features = 91 # 1 + number of days we want to use their price as the training features
print("Dataset size: " + str(number_of_data_sets))
print("Training set size: " + str(number_of_training_sets))
print("Feature set size: " + str(number_of_training_features))

(excel_files_list, processed_excel_files_list) = excel_files_list_processor(excel_files_path, number_of_data_sets)
# stock_list = get_stock_list_from_excel_files(excel_files_path, stock_list_file_path, excel_files_list)
stock_list = get_stock_list_from_processed_file(stock_list_file_path)
# price_list = get_price_list_from_excel_files(stock_list, excel_files_path, price_list_file_path)
price_list = get_price_list_from_processed_file(price_list_file_path, len(stock_list), number_of_data_sets)
# price_list = get_price_list_from_excel_files(stock_list, excel_files_path, price_list_file_path)
price_list = get_price_list_from_processed_file(price_list_file_path, len(stock_list), number_of_data_sets)

prediction_day = 90
price_next_day = [[0 for pd in range(prediction_day)] for si in range(len(stock_list))]
percentage_next_day = [[0. for pd in range(prediction_day)] for si in range(len(stock_list))]
current_price = [0 for si in range(len(stock_list))]

# predicting future prices of a specific stock
# stock_index = np.where(np.array(stock_list) == "وتجارت")[0][0]
# (price_next_day[stock_index], current_price[stock_index]) = linear_regression_NE\
#         (stock_index, price_list, number_of_training_sets, number_of_training_features, number_of_data_sets,
#                   processed_excel_files_list, prediction_day, 5, "ON")

# analysing my stocks
print("\n" + "Analysing My Current Stocks")
my_stocks = get_stock_list_from_processed_file(my_stocks_file_path)
for sn in my_stocks:
    stock_index = np.where(np.array(my_stocks) == sn)[0][0]
    (price_next_day[stock_index], current_price[stock_index]) = linear_regression_NE\
        (stock_index, price_list, number_of_training_sets, number_of_training_features, number_of_data_sets,
                  processed_excel_files_list, prediction_day, 5, "OFF")
    for pd in range(prediction_day):
        temp1 = int(float(price_next_day[stock_index][pd]))
        temp2 = int(float(current_price[stock_index]))
        temp3 = temp1 - temp2
        temp3 = temp3/temp1
        temp3 = temp3*100
        percentage_next_day[stock_index][pd] = math.ceil(temp3*100)/100
    print(my_stocks[stock_index] + ": 1D: " + str(percentage_next_day[stock_index][0]) +
          " | 7D: " + str(percentage_next_day[stock_index][6]) +
          " | 30D: " + str(percentage_next_day[stock_index][29]) +
          " | 60D: " + str(percentage_next_day[stock_index][59]) +
          " | 90D: " + str(percentage_next_day[stock_index][89]))


# analysing all stocks
# print("\n" + "Analysing All Stocks")
# for sn in tqdm(stock_list, desc="Loading…", ascii=False, ncols=75):
#     stock_index = np.where(np.array(stock_list) == sn)[0][0]
#     (price_next_day[stock_index], current_price[stock_index]) = linear_regression_NE\
#         (stock_index, price_list, number_of_training_sets, number_of_training_features, number_of_data_sets,
#                   processed_excel_files_list, prediction_day, 5, "OFF")




