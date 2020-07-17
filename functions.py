import re
import xlrd
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import time

def GET_STOCK_LIST_FROM_EXCEL_FILES(efp, slfp, efl):
    print("Extracting Stock List from Excel Files. Please Wait...")
    # generating stock list
    sl = ["وتجارت"]
    file_index = 0
    for excel_file in efl:
        print("Precessing " + str(file_index + 1) + "/" + str(len(efl)) + " files")
        file_index += 1
        excel_sheet = xlrd.open_workbook(efp + "/" + excel_file).sheet_by_index(0)
        work_book = [""] * (excel_sheet.nrows - 3)
        for row in range(3, excel_sheet.nrows):
            work_book[row - 3] = str(excel_sheet.cell_value(row, 0))
        for stock_name in work_book:
            if stock_name not in sl:
                sl.append(stock_name)
    # writing stock list into file
    open(slfp, 'w').close()
    stock_list_file = open(slfp, "w", encoding="utf8")
    for stock_name_index in range(0, len(sl)):
        stock_list_file.write(sl[stock_name_index])
        if stock_name < len(sl) - 1:
            stock_list_file.write("\n")
    stock_list_file.close()
    print("Done!")
    return sl

def GET_STOCK_LIST_FROM_PROCESSED_FILE(slfp):
    #print("Loading Stock List. Please Wait...")
    # reading processed file
    with open(slfp, encoding="utf8") as data:
        stock_list = []
        for line in data:
            line = re.sub(" ", "_", str(line))
            part = line.split()
            stock_list.append(str(part[0]))
    #print("Done!")
    return stock_list

def BUILD_PRELIMINARY_FILE(file_name, number_of_rows):
    file = open(file_name, "w")
    for row in range(0, number_of_rows):
        file.write("xxx")
        if row < number_of_rows - 1:
            file.write("\n")
    file.close()

def GET_WORK_BOOK_INFO(file_name):
    excel_file = xlrd.open_workbook(file_name)
    sheet = excel_file.sheet_by_index(0)
    wb_stocks = [""] * (sheet.nrows - 3)
    wb_prices = [0] * (sheet.nrows - 3)
    # reading stock names and prices
    for r in range(3, sheet.nrows):
        wb_stocks[r - 3] = re.sub(" ", "_", str(sheet.cell_value(r, 0)))
        wb_prices[r - 3] = sheet.cell_value(r, 10)
    return wb_stocks, wb_prices

def BUILD_PRICES_FILE(file_name, name, price):
    line_index = 0
    new_line = [""] * len(name)
    # generating lines
    with open(file_name) as data:
        for line in data:
            line = line.rstrip("\n")
            if line == "xxx":
                new_line[line_index] = str(price[line_index])
            else:
                new_line[line_index] = str(line) + "\t" + str(price[line_index])
            line_index += 1
    # writing lines
    open(file_name, 'w').close()
    prices_file = open(file_name, "w")
    for stock_name_index in range(0, len(name)):
        prices_file.write(new_line[stock_name_index])
        if stock_name_index < len(name) - 1:
            prices_file.write("\n")
    prices_file.close()

def GET_PRICE_LIST_FROM_EXCEL_FILES(sl, efp, plfp):
    print("Generating Prices File. Please Wait...")
    excel_files_list = [f for f in listdir(efp) if isfile(join(efp, f))]
    price_list = [[0 for x in range(len(excel_files_list))] for y in range(len(sl))]
    last_valid_price = [0 for x in range(len(sl))]
    BUILD_PRELIMINARY_FILE(plfp, len(sl))
    # generating price list
    excel_file_index = 0
    for excel_file in tqdm(excel_files_list, desc="Loading…", ascii=False, ncols=75):
        current_price = [0] * len(sl)
        (wb_stocks, wb_prices) = GET_WORK_BOOK_INFO(efp + "/" + excel_file)
        for stock_name in sl:
            if stock_name in wb_stocks:
                if wb_prices[wb_stocks.index(stock_name)] != 0:
                    current_price[sl.index(stock_name)] = wb_prices[wb_stocks.index(stock_name)]
                    last_valid_price[sl.index(stock_name)] = wb_prices[wb_stocks.index(stock_name)]
                elif wb_prices[wb_stocks.index(stock_name)] == 0 and last_valid_price[sl.index(stock_name)] != 0:
                    current_price[sl.index(stock_name)] = last_valid_price[sl.index(stock_name)]
                elif wb_prices[wb_stocks.index(stock_name)] == 0 and last_valid_price[sl.index(stock_name)] == 0:
                    current_price[sl.index(stock_name)] = 0
            else:
                current_price[sl.index(stock_name)] = last_valid_price[sl.index(stock_name)]
        temp_stock_index = 0
        for stock_price in price_list:
            stock_price[excel_file_index] = current_price[temp_stock_index]
            temp_stock_index += 1
        excel_file_index += 1
        BUILD_PRICES_FILE(plfp, sl, current_price)
    print("Done!")
    return price_list

def GET_PRICE_LIST_FROM_PROCESSED_FILE(plfp, ns, nds):
    #print("Loading Price List. Please Wait...")
    with open(plfp, encoding="utf8") as data:
        price_list = [[0 for x in range(nds)] for y in range(ns)]
        stock_name_index = 0
        for line in data:
            line = re.sub(" ", "_", str(line))
            part = line.split()
            for data_set_index in range(0, nds):
                price_list[stock_name_index][data_set_index] = part[data_set_index]
            stock_name_index += 1
    #print("Done!")
    return price_list

def NORMALIZE_FEATURES(X, number_of_rows, number_of_columns, m, s):
    normal_X = np.zeros(shape=(number_of_rows, number_of_columns))
    mean = [0 for m in range(0, number_of_columns)]
    std = [0 for m in range(0, number_of_columns)]
    normal_X[:, 0] = X[:, 0]
    for col in range(1, number_of_columns):
        mean[col] = np.mean(X, axis=0)[col]
        std[col] = np.std(X, axis=0, ddof=1)[col]
        if std[col] == 0:
            std[col] = 1
        normal_X[:, col] = (X[:, col] - mean[col]) / std[col]
    return normal_X, mean, std

def LINEAR_REGRESSION_USING_NORMAL_EQUATION(stock_id, pl, ntds, ntf, nds, pefl, ntd, regularization, demonstration):
    if demonstration == "ON":
        print("\n" + "Linear Regression is ON!")
    # building training data
    regularization_matrix = np.zeros((ntf, ntf), float)
    np.fill_diagonal(regularization_matrix, regularization)
    regularization_matrix[0][0] = 0
    training_X = [[0 for col in range(0, ntf)] for row in range(0, ntds)]
    training_Y = [0 for row in range(0, ntds)]
    for row in range(0, ntds):
        training_X[row][0] = 1
        for col in range(1, ntf):
            training_X[row][col] = int(float(pl[stock_id][row + col - 1]))
    for row in range(0, ntds):
        training_Y[row] = int(float(pl[stock_id][row + ntf - 1]))
    BUILD_OCTAVE_TEST_FILE(training_X, training_Y, ntds, ntf)
    training_X = np.array(training_X)
    training_Y = np.array(training_Y)
    normal_training_X, mean, std = NORMALIZE_FEATURES(training_X, ntds, ntf)
    # calculating learning coefficients
    temp1 = np.matmul(normal_training_X.transpose(), normal_training_X)
    temp2 = np.matmul(np.linalg.pinv(temp1 + regularization_matrix), normal_training_X.transpose())
    theta = np.matmul(temp2, training_Y)
    # generating predicted Y for training data
    training_predicted_Y = np.matmul(normal_training_X, theta)
    # demonstrating training phase
    if demonstration == "ON":
        print("MSE on training set: " + str(MSE(training_predicted_Y, training_Y)))
        print("MAE on training set: " + str(MAE(training_predicted_Y, training_Y)))
        PLOT_DIAGRAM(pefl, training_Y, training_predicted_Y, 0, 0)



    # test phase
    if prediction_day > 0:
        price_next_day = [0 for row in range(0, prediction_day)]
        X_next_day = [1 for col in range(0, number_of_feature_columns)]
        normal_X_next_day = [1 for col in range(0, number_of_feature_columns)]
        for col in range(1, number_of_feature_columns):
            X_next_day[col] = int(float(price_list[stock_id][number_of_dataset_rows - number_of_feature_columns + col]))

        for pd in range(0, prediction_day):
            for col in range(1, number_of_feature_columns):
                normal_X_next_day[col] = (X_next_day[col] - mean[col]) / std[col]
            price_next_day[pd] = math.ceil(np.matmul(normal_X_next_day, theta))

            X_next_day = np.roll(X_next_day, -1)
            X_next_day[number_of_feature_columns-1] = price_next_day[pd]
            X_next_day[0] = 1

        if demonstration == "ON":
            print("Expected price of next day: " + str(price_next_day[0]))
            prediction_diagram(price_next_day, price_list[stock_id][number_of_dataset_rows-1])

    return price_next_day, price_list[stock_id][number_of_dataset_rows-1]


def BUILD_OCTAVE_TEST_FILE(X, Y, number_of_rows, number_of_columns):
    file_name = "text_files/test.txt"
    open("text_files/test.txt", 'w').close()
    # generating lines
    line_index = 0
    new_line = [""] * number_of_rows
    for row in range(0, number_of_rows):
        for col in range(1, number_of_columns):
            new_line[row] = new_line[row] + str(X[row][col]) + ","
        new_line[row] = new_line[row] + str(Y[row])
    # writing file
    file = open(file_name, "w")
    for row in range(0, number_of_rows):
        file.write(new_line[row])
        if row < number_of_rows - 1:
            file.write("\n")
    file.close()


def MSE(predicted_Y, Y): # mean squared error
    return np.dot((predicted_Y - Y), (predicted_Y - Y))\
           /(2 * len(Y))


def MAE(predicted_Y, Y): # mean absolute error
    return np.sum(np.abs(predicted_Y - Y))/(2 * len(Y))


def PROCESS_EXCEL_FILES_LIST(efp, nds):
    efl = [f for f in listdir(efp) if isfile(join(efp, f))]
    label = np.arange(0, nds, 1)
    pefl = [""] * len(label)
    # generating lists
    for file_name_index in range(0, len(label)):
        pefl[file_name_index] = efl[label[file_name_index]]
    for fn in range(0, len(pefl)):
        pefl[fn] = re.sub('.xlsx', "", re.sub('_', "", re.sub('MarketWatchPlus-13', "", pefl[fn])))
    return efl, pefl


def PLOT_DIAGRAM(pefl, Y, predicted_Y, label_index, ntd):
    #  generating x axis label for prediction phase
    if ntd > 0:
        extra_days = ["" for ed in range(ntd)]
        for day in range(ntd):
            extra_days[day] = "day+" + str(day + 1)
        pefl =  pefl + extra_days
    pefl = np.array(pefl)
    temp_efl =np.split(pefl, [label_index, len(Y) + label_index])[1]
    # plotting diagram
    label = np.arange(label_index, label_index + len(Y), 1)
    plt.plot(label, Y, 'r--', label, predicted_Y, 'b*')
    plt.xticks(label, temp_efl, rotation='vertical')
    plt.grid()
    plt.show()


def prediction_diagram(price_next_day, price):
    X_label = np.arange(0, len(price_next_day), 1)
    last_price_line = [int(float(price)) for ed in range(len(price_next_day))]
    extra_days = ["" for ed in range(len(price_next_day))]
    for ed in range(0, len(price_next_day)):
        extra_days[ed] = "day+" + str(ed + 1)

    plt.plot(X_label, last_price_line, 'r--', price_next_day, 'b*')
    plt.xticks(X_label, extra_days, rotation='vertical')

    ax = plt.gca()
    ax.grid(axis='both', which='both')
    plt.show()


def CL(string, length): # add spaces at the end of given string to make its length equal to the given length
    if len(string) < length:
        for i in range(length-len(string)):
            string = string + " "
    return string


def ANALYZE_MY_STOCKS(sl, msfp, msenfp, pl, ntds, ntf, nds, pefl, msafp, ntd):
    print("\n" + "Analysing My Current Stocks")
    next_day_prices = [[0 for pd in range(ntd)] for si in range(len(sl))]
    next_day_per = [[0. for pd in range(ntd)] for si in range(len(sl))] # per: percentages
    current_price = [0 for si in range(len(sl))]
    my_stocks = GET_STOCK_LIST_FROM_PROCESSED_FILE(msfp)
    my_stocks_en = GET_STOCK_LIST_FROM_PROCESSED_FILE(msenfp)
    for stock_name in tqdm(my_stocks, desc="Loading…", ascii=False, ncols=75):
        stock_index = np.where(np.array(my_stocks) == stock_name)[0][0]
        (next_day_prices[stock_index], current_price[stock_index]) = \
            LINEAR_REGRESSION_USING_NORMAL_EQUATION(stock_index, pl, ntds, ntf, nds, pefl, ntd, 5, "OFF")

        VALIDATE_TRAINING_COEFS_OF_LRNE(pefl, nds, ntds, ntf, pl, stock_index, mean, std, theta, "OFF")


        for day in range(ntd):
            temp1 = int(float(next_day_prices[stock_index][day]))
            temp2 = int(float(current_price[stock_index]))
            temp3 = temp1 - temp2
            temp3 = temp3 / temp1
            temp3 = temp3 * 100
            next_day_per[stock_index][day] = math.ceil(temp3 * 100) / 100

    open(msafp, 'w').close()
    ms_analysis_file = open(msafp, "w")
    ms_analysis_file.write("------------------------------------------------------------------------" + "\n")
    ms_analysis_file.write("|NAME           |1 DAY     |7 DAY     |1 Month   |2 Month   |3 Month   |" + "\n")
    ms_analysis_file.write("------------------------------------------------------------------------" + "\n")
    for sn in my_stocks:
        si = np.where(np.array(my_stocks) == sn)[0][0]
        ms_analysis_file.write("|" + CL(my_stocks_en[si], 15)
                               + "|" + CL(str(next_day_per[si][0]), 10)
                               + "|" + CL(str(next_day_per[si][6]), 10)
                               + "|" + CL(str(next_day_per[si][29]), 10)
                               + "|" + CL(str(next_day_per[si][59]), 10)
                               + "|" + CL(str(next_day_per[si][89]), 10) + "\n")
        ms_analysis_file.write("------------------------------------------------------------------------" + "\n")
    ms_analysis_file.close()

    time.sleep(0.5)
    print("Analysis File is Built.")


def VALIDATE_LRNE_DBD(pefl, nds, ntds, ntf, pl, stock_id, mean, std, theta, demonstration): # DBD: day by day
    # validation phase
    validation_Y = [0 for row in range(0, nds - (ntds + ntf - 1))]
    validation_X = [[0 for col in range(0, ntf)] for row in range(0, nds - (ntds + ntf - 1))]
    normal_validation_X = np.zeros(shape=(nds - (ntds + ntf - 1), ntf))
    # building validation data
    for row in range(0, len(validation_Y)):
        if (ntds + ntf - 1 + row) < nds:
            validation_Y[row] = int(float(pl[stock_id][ntds + ntf - 1 + row]))
    for row in range(0, len(validation_X)):
        validation_X[row][0] = 1
        for col in range(1, ntf):
            validation_X[row][col] = int(float(pl[stock_id][row + ntds + ntf - 1 - ntf + col]))
    validation_X = np.array(validation_X)
    validation_Y = np.array(validation_Y)
    # normalizing
    normal_validation_X[:, 0] = validation_X[:, 0]
    for col in range(1, ntf):
        normal_validation_X[:, col] = (validation_X[:, col] - mean[col]) / std[col]
    # generating predicted Y for validation data
    validation_predicted_Y = np.matmul(normal_validation_X, theta)
    # demonstrating validation phase
    if demonstration == "ON":
        print("MSE on validation set: " + str(MSE(validation_predicted_Y, validation_Y)))
        print("MAE on validation set: " + str(MAE(validation_predicted_Y, validation_Y)))
        PLOT_DIAGRAM(pefl, validation_Y, validation_predicted_Y, ntds + ntf - 1, 0)

def VALIDATE_LRNE_LT(pefl, nds, ntds, ntf, pl, stock_id, mean, std, theta, demonstration): # LT: long term
    # validation phase
    validation_X = [1 for col in range(0, ntf)]
    normal_validation_X = [1 for col in range(0, ntf)]
    price_next_day = [0 for row in range(0, nds - ntds)]
    validation_Y = [0 for row in range(0, nds - ntds)]
    # building validation data
    for row in range(0, len(validation_Y)):
        validation_Y[row] = int(float(pl[stock_id][ntds + 1 + row]))
    for col in range(1, ntf):
        validation_X[col] = int(float(pl[stock_id][ntds - ntf + col]))
    # calculating next day prices
    for day in range(0, len(price_next_day)):
        for col in range(1, ntf):
            normal_validation_X[col] = (validation_X[col] - mean[col]) / std[col]
        price_next_day[day] = math.ceil(np.matmul(normal_validation_X, theta))
        validation_X = np.roll(validation_X, -1)
        validation_X[ntf - 1] = price_next_day[day]
        validation_X[0] = 1

    if demonstration == "ON":
        print("Expected price of next day: " + str(price_next_day[0]))
        prediction_diagram(price_next_day, price_list[stock_id][number_of_dataset_rows - 1])







    validation_X = np.array(validation_X)
    validation_Y = np.array(validation_Y)
    # normalizing
    normal_validation_X[:, 0] = validation_X[:, 0]
    for col in range(1, ntf):
        normal_validation_X[:, col] = (validation_X[:, col] - mean[col]) / std[col]
    # generating predicted Y for validation data
    validation_predicted_Y = np.matmul(normal_validation_X, theta)
    # demonstrating validation phase
    if demonstration == "ON":
        print("MSE on validation set: " + str(MSE(validation_predicted_Y, validation_Y)))
        print("MAE on validation set: " + str(MAE(validation_predicted_Y, validation_Y)))
        PLOT_DIAGRAM(pefl, validation_Y, validation_predicted_Y, ntds + ntf - 1, 0)