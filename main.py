from functions import *

# initializing input parameters
efp = "excel_files" # excel_files_path
slfp = "text_files/stock_list.txt" # stock_list_file_path
msfp = "text_files/my_stocks.txt" # my_stocks_file_path
msenfp = "text_files/my_stocks_en.txt" # my_stocks_en_file_path
msafp = "text_files/my_stocks_analysis.txt" # my_stocks_analysis_file_path
plfp = "text_files/price_list.txt" # price_list_file_path
nds = 311 # number_of_data_sets
ntds = 200 # number_of_training_sets
ntf = 91 # number_of_training_features, 1 + number of days we want to use their price as the training features

# introducing the size of input sets
print("Dataset size: " + str(nds))
print("Training set size: " + str(ntds))
print("Feature set size: " + str(ntf))

(efl, pefl) = PROCESS_EXCEL_FILES_LIST(efp, nds) # efl: excel_files_list, pefl: processed_excel_files_list
# sl = GET_STOCK_LIST_FROM_EXCEL_FILES(efp, slfp, efl) # sl: stock_list
sl = GET_STOCK_LIST_FROM_PROCESSED_FILE(slfp) # sl: stock_list
ns = len(sl) # ns: number of stocks
# pl = GET_PRICE_LIST_FROM_EXCEL_FILES(sl, efp, plfp) # pl: price_list
pl = GET_PRICE_LIST_FROM_PROCESSED_FILE(plfp, ns, nds) # pl: price_list

# analysing my stocks
ntd = 90 # number of days you need to predict the prices and percentages
print("\n" + "Analysing My Current Stocks")
ndp = [[0 for x in range(ntd)] for y in range(len(sl))] # ndp: next_day_prices
ndd = [[0. for x in range(ntd)] for y in range(len(sl))] # ndd: next_day_percentages
cp = [0 for x in range(len(sl))] # cp: current_price
ms = GET_STOCK_LIST_FROM_PROCESSED_FILE(msfp) # ms: my_stocks
mse = GET_STOCK_LIST_FROM_PROCESSED_FILE(msenfp) # mse: my_stocks_en


si = np.where(np.array(ms) == "وتجارت")[0][0] # si: stock_index
mean, std, theta = LEARN_BY_NELR(si, pl, ntds, ntf, pefl, 5, "OFF")
VALIDATE_LRNE_DBD(pefl, nds, ntds, ntf, pl, si, mean, std, theta, "OFF")
VALIDATE_LRNE_LT(pefl, nds, ntds, ntf, pl, si, mean, std, theta, "ON")
(ndp[si], cp[si]) = PREDICT_LRNE(ntd, ntf, ntds+ntf-1, pl, si, mean, std, theta, "ON")



# for sn in tqdm(ms, desc="Loading…", ascii=False, ncols=75): # sn: stock_name
#     si = np.where(np.array(ms) == sn)[0][0] # si: stock_index
#     mean, std, theta = LEARN_BY_NELR(si, pl, ntds, ntf, pefl, 5, "OFF")
#     VALIDATE_LRNE_DBD(pefl, nds, ntds, ntf, pl, si, mean, std, theta, "OFF")
#     VALIDATE_LRNE_LT(pefl, nds, ntds, ntf, pl, si, mean, std, theta, "ON")
#     (ndp[si], cp[si]) = PREDICT_LRNE(ntd, ntf, nds, pl, si, mean, std, theta, "ON")
#     # calculating progress percentages
#     for d in range(ntd): # d: day
#         temp1 = int(float(ndp[si][d]))
#         temp2 = int(float(cp[si]))
#         temp3 = temp1 - temp2
#         temp3 = temp3 / temp1
#         temp3 = temp3 * 100
#         ndd[si][d] = math.ceil(temp3 * 100) / 100
# # writing results in file
# open(msafp, 'w').close()
# maf = open(msafp, "w") # maf: ms_analysis_file
# maf.write("------------------------------------------------------------------------" + "\n")
# maf.write("|NAME           |1 DAY     |7 DAY     |1 Month   |2 Month   |3 Month   |" + "\n")
# maf.write("------------------------------------------------------------------------" + "\n")
# for sn in ms:
#     si = np.where(np.array(ms) == sn)[0][0]
#     maf.write("|" + CORRECT_LENGTH(mse[si], 15)
#                            + "|" + CORRECT_LENGTH(str(ndd[si][0]), 10)
#                            + "|" + CORRECT_LENGTH(str(ndd[si][6]), 10)
#                            + "|" + CORRECT_LENGTH(str(ndd[si][29]), 10)
#                            + "|" + CORRECT_LENGTH(str(ndd[si][59]), 10)
#                            + "|" + CORRECT_LENGTH(str(ndd[si][89]), 10) + "\n")
#     maf.write("------------------------------------------------------------------------" + "\n")
# maf.close()
# time.sleep(0.5)
# print("Analysis File is Built.")



















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




