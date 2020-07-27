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
vs = EXTRACT_VALID_STOCKS(pl, nds) # vs: valid_stocks

# analysing a specific stock
sn = "خصدرا"
esn = "Z7022" # esn: english_stock_name
print("\n" + "Analysing " + esn + ":")
ntd = 90 # number of days you need to predict the prices and percentages
ndp = [0 for x in range(ntd)] # ndp: next_day_prices
ndd = [0. for x in range(ntd)] # ndd: next_day_percentages
cp = 0 # cp: current_price
si = np.where(np.array(sl) == sn)[0][0]  # si: stock_index
mean, std, theta = LEARN_BY_NELR(si, esn, pl, ntds, ntf, pefl, 5, "ON")
VALIDATE_LRNE_DBD(pefl, nds, ntds, ntf, pl, si, esn, mean, std, theta, "ON")
VALIDATE_LRNE_LT(pefl, nds, ntds, ntf, pl, si, esn, mean, std, theta, "ON")
(ndp, cp) = PREDICT_LRNE(ntd, ntf, nds, pl, si, esn, mean, std, theta, "ON")
# calculating progress percentages
for d in range(ntd): # d: day
    temp1 = int(float(ndp[d]))
    temp2 = int(float(cp))
    temp3 = temp1 - temp2
    temp3 = temp3 / temp2
    temp3 = temp3 * 100
    ndd[d] = math.ceil(temp3 * 100) / 100
# showing results
print("Current price: " + str(cp))
print("Expected price of next day: " + str(ndp[0]))
print("Next day percentage: " + str(ndd[0]) + "%")
print("Next week percentage: " + str(ndd[6]) + "%")
print("Next month percentage: " + str(ndd[29]) + "%")

# # analysing my stocks
# ntd = 90 # number of days you need to predict the prices and percentages
# print("\n" + "Analysing my current stocks")
# ndp = [[0 for x in range(ntd)] for y in range(len(sl))] # ndp: next_day_prices
# ndd = [[0. for x in range(ntd)] for y in range(len(sl))] # ndd: next_day_percentages
# cp = [0 for x in range(len(sl))] # cp: current_price
# ms = GET_STOCK_LIST_FROM_PROCESSED_FILE(msfp) # ms: my_stocks
# mse = GET_STOCK_LIST_FROM_PROCESSED_FILE(msenfp) # mse: my_stocks_en
# for sn in tqdm(ms, desc="Loading…", ascii=False, ncols=75): # sn: stock_name
#     gsi = np.where(np.array(sl) == sn)[0][0] # gsi: general_stock_index
#     csi = np.where(np.array(ms) == sn)[0][0] # csi: current_stock_index
#     mean, std, theta = LEARN_BY_NELR(gsi, mse[csi], pl, ntds, ntf, pefl, 5, "OFF")
#     VALIDATE_LRNE_DBD(pefl, nds, ntds, ntf, pl, gsi, mse[csi], mean, std, theta, "OFF")
#     VALIDATE_LRNE_LT(pefl, nds, ntds, ntf, pl, gsi, mse[csi], mean, std, theta, "OFF")
#     (ndp[gsi], cp[gsi]) = PREDICT_LRNE(ntd, ntf, nds, pl, gsi, mse[csi], mean, std, theta, "OFF")
#     # calculating progress percentages
#     for d in range(ntd): # d: day
#         temp1 = int(float(ndp[gsi][d]))
#         temp2 = int(float(cp[gsi]))
#         temp3 = temp1 - temp2
#         temp3 = temp3 / temp2
#         temp3 = temp3 * 100
#         ndd[gsi][d] = math.ceil(temp3 * 100) / 100
# # writing results in file
# open(msafp, 'w').close()
# maf = open(msafp, "w") # maf: ms_analysis_file
# maf.write("------------------------------------------------------------------------" + "\n")
# maf.write("|NAME           |1 DAY     |7 DAY     |1 Month   |2 Month   |3 Month   |" + "\n")
# maf.write("------------------------------------------------------------------------" + "\n")
# for sn in ms:
#     gsi = np.where(np.array(sl) == sn)[0][0]  # gsi: general_stock_index
#     csi = np.where(np.array(ms) == sn)[0][0]  # csi: current_stock_index
#     maf.write("|" + CORRECT_LENGTH(mse[csi], 15)
#                            + "|" + CORRECT_LENGTH(str(ndd[gsi][0]), 10)
#                            + "|" + CORRECT_LENGTH(str(ndd[gsi][6]), 10)
#                            + "|" + CORRECT_LENGTH(str(ndd[gsi][29]), 10)
#                            + "|" + CORRECT_LENGTH(str(ndd[gsi][59]), 10)
#                            + "|" + CORRECT_LENGTH(str(ndd[gsi][89]), 10) + "\n")
#     maf.write("------------------------------------------------------------------------" + "\n")
# maf.close()
# time.sleep(0.5)
# print("Analysis file is built.")

# # finding the most profitable stocks
# ntd = 90 # number of days you need to predict the prices and percentages
# print("\n" + "Finding the most profitable stocks")
# ndp = [[0 for x in range(ntd)] for y in range(len(sl))] # ndp: next_day_prices
# ndd = [[0. for x in range(ntd)] for y in range(len(sl))] # ndd: next_day_percentages
# cp = [0 for x in range(len(sl))] # cp: current_price
# for sn in tqdm(sl, desc="Loading…", ascii=False, ncols=75): # sn: stock_name
#     si = np.where(np.array(sl) == sn)[0][0]
#     mean, std, theta = LEARN_BY_NELR(si, sl[si], pl, ntds, ntf, pefl, 5, "OFF")
#     #VALIDATE_LRNE_DBD(pefl, nds, ntds, ntf, pl, si, sl[si], mean, std, theta, "OFF")
#     #VALIDATE_LRNE_LT(pefl, nds, ntds, ntf, pl, si, sl[si], mean, std, theta, "OFF")
#     (ndp[si], cp[si]) = PREDICT_LRNE(ntd, ntf, nds, pl, si, sl[si], mean, std, theta, "OFF")
#     # calculating progress percentages
#     for d in range(ntd): # d: day
#         temp1 = int(float(ndp[si][d]))
#         temp2 = int(float(cp[si]))
#         temp3 = temp1 - temp2
#         temp3 = temp3 / temp2
#         temp3 = temp3 * 100
#         ndd[si][d] = math.ceil(temp3 * 100) / 100
# ndd = np.array(ndd)
# print("Calculation is done!")
# # finding the best stocks of the next day
# print("\n" + "Best stocks of tomorrow:")
# for bsi in range(10): # bsi: best_stock_index
#     flag = True
#     while flag == True:
#         mi = np.argmax(ndd[:, 0])
#         if vs[mi] == 1:
#             flag = False
#             print(sl[np.argmax(ndd[:, 0])] + " (" + str(ndd[:, 0][np.argmax(ndd[:, 0])]) + ")")
#             ndd[:, 0][np.argmax(ndd[:, 0])] = 0
#         else:
#             ndd[:, 0][np.argmax(ndd[:, 0])] = 0
# # finding the best stocks of the next week
# print("\n" + "Best stocks of the next week:")
# for bsi in range(10): # bsi: best_stock_index
#     flag = True
#     while flag == True:
#         mi = np.argmax(ndd[:, 6])
#         if vs[mi] == 1:
#             flag = False
#             print(sl[np.argmax(ndd[:, 6])] + " (" + str(ndd[:, 6][np.argmax(ndd[:, 6])]) + ")")
#             ndd[:, 6][np.argmax(ndd[:, 6])] = 0
#         else:
#             ndd[:, 6][np.argmax(ndd[:, 6])] = 0
# # finding the best stocks of the next week
# print("\n" + "Best stocks of the next month:")
# for bsi in range(10): # bsi: best_stock_index
#     flag = True
#     while flag == True:
#         mi = np.argmax(ndd[:, 29])
#         if vs[mi] == 1:
#             flag = False
#             print(sl[np.argmax(ndd[:, 29])] + " (" + str(ndd[:, 29][np.argmax(ndd[:, 29])]) + ")")
#             ndd[:, 29][np.argmax(ndd[:, 29])] = 0
#         else:
#             ndd[:, 29][np.argmax(ndd[:, 29])] = 0
# print("\n" + "Best stocks of the next 2 month:")
# for bsi in range(10): # bsi: best_stock_index
#     flag = True
#     while flag == True:
#         mi = np.argmax(ndd[:, 59])
#         if vs[mi] == 1:
#             flag = False
#             print(sl[np.argmax(ndd[:, 59])] + " (" + str(ndd[:, 59][np.argmax(ndd[:, 59])]) + ")")
#             ndd[:, 59][np.argmax(ndd[:, 59])] = 0
#         else:
#             ndd[:, 59][np.argmax(ndd[:, 59])] = 0
# print("\n" + "Best stocks of the next 3 month:")
# for bsi in range(10): # bsi: best_stock_index
#     flag = True
#     while flag == True:
#         mi = np.argmax(ndd[:, 89])
#         if vs[mi] == 1:
#             flag = False
#             print(sl[np.argmax(ndd[:, 89])] + " (" + str(ndd[:, 89][np.argmax(ndd[:, 89])]) + ")")
#             ndd[:, 89][np.argmax(ndd[:, 89])] = 0
#         else:
#             ndd[:, 89][np.argmax(ndd[:, 89])] = 0