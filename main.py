from functions import *

efp = "excel_files" # excel_files_path
slfp = "text_files/stock_list.txt" # stock_list_file_path
msfp = "text_files/my_stocks.txt" # my_stocks_file_path
msenfp = "text_files/my_stocks_en.txt" # my_stocks_en_file_path
msafp = "text_files/my_stocks_analysis.txt" # my_stocks_analysis_file_path
plfp = "text_files/price_list.txt" # price_list_file_path

nds = 311 # number_of_data_sets
ntds = 200 # number_of_training_sets
ntf = 91 # number_of_training_features. 1 + number of days we want to use their price as the training features

print("Dataset size: " + str(nds))
print("Training set size: " + str(ntds))
print("Feature set size: " + str(ntf))

(efl, pefl) = process_excel_files_list(efp, nds) # efl: excel_files_list, pefl: processed_excel_files_list
# sl = get_stock_list_from_excel_files(efp, slfp, efl) # sl: stock_list
sl = get_stock_list_from_processed_file(slfp) # sl: stock_list
# pl = get_price_list_from_excel_files(sl, efp, plfp) # pl: price_list
pl = get_price_list_from_processed_file(plfp, len(sl), nds) # pl: price_list

# analysing my stocks
ntd = 90 # number of days you need to predict the prices and percentages
ANALYZE_MY_STOCKS(sl, msfp, msenfp, pl, ntds, ntf, nds, pefl, msafp, ntd)











prediction_day = 90
price_next_day = [[0 for pd in range(prediction_day)] for si in range(len(sl))]
percentage_next_day = [[0. for pd in range(prediction_day)] for si in range(len(sl))]
current_price = [0 for si in range(len(sl))]


# predicting future prices of a specific stock
# stock_index = np.where(np.array(stock_list) == "وتجارت")[0][0]
# (price_next_day[stock_index], current_price[stock_index]) = linear_regression_NE\
#         (stock_index, price_list, number_of_training_sets, number_of_training_features, number_of_data_sets,
#                   processed_excel_files_list, prediction_day, 5, "ON")



# analysing all stocks
# print("\n" + "Analysing All Stocks")
# for sn in tqdm(stock_list, desc="Loading…", ascii=False, ncols=75):
#     stock_index = np.where(np.array(stock_list) == sn)[0][0]
#     (price_next_day[stock_index], current_price[stock_index]) = linear_regression_NE\
#         (stock_index, price_list, number_of_training_sets, number_of_training_features, number_of_data_sets,
#                   processed_excel_files_list, prediction_day, 5, "OFF")




