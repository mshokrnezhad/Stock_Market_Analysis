from functions import *

excel_files_path = "excel_files"
stock_list_file_path = "text_files/stock_list.txt"
price_list_file_path = "text_files/price_list.txt"
number_of_data_sets = 68 # we have 68 columns of data.
number_of_training_sets = 40 # we decide to use 40 sets of data to train the h function.
number_of_training_features = 21 # 1 + number of days we want to use their price as the training features

print("Dataset size: " + str(number_of_data_sets))
print("Training set size: " + str(number_of_training_sets))
print("Feature set size: " + str(number_of_training_features))

(excel_files_list, processed_excel_files_list) = excel_files_list_processor(excel_files_path, number_of_data_sets)
# stock_list = get_stock_list_from_excel_files(excel_files_path, stock_list_file_path, excel_files_list)
stock_list = get_stock_list_from_processed_file(stock_list_file_path)
# price_list = get_price_list_from_excel_files(stock_list, excel_files_path, price_list_file_path)
price_list = get_price_list_from_processed_file(price_list_file_path, len(stock_list), number_of_data_sets)

linear_regression_NE(0, price_list, number_of_training_sets, number_of_training_features, number_of_data_sets,
                  processed_excel_files_list)



