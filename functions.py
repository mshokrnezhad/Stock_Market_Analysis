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
    fi = 0 # fi: file_index
    for ef in efl: # ef: excel_file
        print("Precessing " + str(fi + 1) + "/" + str(len(efl)) + " files")
        fi += 1
        es = xlrd.open_workbook(efp + "/" + ef).sheet_by_index(0) # es: excel_sheet
        wb = [""] * (es.nrows - 3) # wb: workbook
        for row in range(3, es.nrows):
            wb[row - 3] = str(es.cell_value(row, 0))
        for sn in wb: # sn: stock_name
            if sn not in sl:
                sl.append(sn)
    # writing stock list into file
    open(slfp, 'w').close()
    slf = open(slfp, "w", encoding="utf8") # slf: stock_list_file
    for sni in range(0, len(sl)): # sni: stock_name_index
        slf.write(sl[sni])
        if sni < len(sl) - 1:
            slf.write("\n")
    slf.close()
    print("Done!")
    return sl

def GET_STOCK_LIST_FROM_PROCESSED_FILE(slfp):
    #print("Loading Stock List. Please Wait...")
    # reading processed file
    with open(slfp, encoding="utf8") as data:
        sl = [] #sl: stock_list
        for l in data: # l: line
            l = re.sub(" ", "_", str(l))
            sl.append(str(l.split()[0]))
    #print("Done!")
    return sl

def BUILD_PRELIMINARY_FILE(fn, nr):
    f = open(fn, "w") # f: file
    for r in range(0, nr): # r: row
        f.write("xxx")
        if r < nr - 1:
            f.write("\n")
    f.close()

def GET_WORK_BOOK_INFO(fn):
    ef = xlrd.open_workbook(fn) # ef: excel_file
    s = ef.sheet_by_index(0) # s: sheet
    wbs = [""] * (s.nrows - 3) # wbs: wb_stocks
    wbp = [0] * (s.nrows - 3) # wbp: wb_prices
    # reading stock names and prices
    for r in range(3, s.nrows): # r: row
        wbs[r - 3] = re.sub(" ", "_", str(s.cell_value(r, 0)))
        wbp[r - 3] = s.cell_value(r, 10)
    return wbs, wbp

def BUILD_PRICES_FILE(fn, n, p):
    li = 0 # li: line_index
    nl = [""] * len(n) # nl: new_line
    # generating lines
    with open(fn) as data:
        for l in data: # l: line
            l = l.rstrip("\n")
            if l == "xxx":
                nl[li] = str(p[li])
            else:
                nl[li] = str(l) + "\t" + str(p[li])
            li += 1
    # writing lines
    open(fn, 'w').close()
    pf = open(fn, "w") # pf: prices_file
    for sni in range(0, len(n)): # sni: stock_name_index
        pf.write(nl[sni])
        if sni < len(n) - 1:
            pf.write("\n")
    pf.close()

def GET_PRICE_LIST_FROM_EXCEL_FILES(sl, efp, plfp):
    print("Generating Prices File. Please Wait...")
    efl = [f for f in listdir(efp) if isfile(join(efp, f))] # efl: excel_files_list
    pl = [[0 for x in range(len(efl))] for y in range(len(sl))] # pl: price_list
    lvp = [0 for x in range(len(sl))] # lvp: last_valid_price
    BUILD_PRELIMINARY_FILE(plfp, len(sl))
    # generating price list
    efi = 0 # efi: excel_file_index
    for ef in tqdm(efi, desc="Loading…", ascii=False, ncols=75): # ef: excel_file
        cp = [0] * len(sl) # cp: current_price
        (wbs, wbp) = GET_WORK_BOOK_INFO(efp + "/" + ef) # wbs: wb_stocks, wbp: wb_prices
        for sn in sl: # sn: stock_name
            if sn in wbs:
                if wbp[wbs.index(sn)] != 0:
                    cp[sl.index(sn)] = wbp[wbs.index(sn)]
                    lvp[sl.index(sn)] = wbp[wbs.index(sn)]
                elif wbp[wbs.index(sn)] == 0 and lvp[sl.index(sn)] != 0:
                    cp[sl.index(sn)] = lvp[sl.index(sn)]
                elif wbp[wbs.index(sn)] == 0 and lvp[sl.index(sn)] == 0:
                    cp[sl.index(sn)] = 0
            else:
                cp[sl.index(sn)] = lvp[sl.index(sn)]
        tsi = 0 # tsi: temp_stock_index
        for sp in pl: # sp: stock_price
            sp[efi] = cp[tsi]
            tsi += 1
        efi += 1
        BUILD_PRICES_FILE(plfp, sl, cp)
    print("Done!")
    return pl

def GET_PRICE_LIST_FROM_PROCESSED_FILE(plfp, ns, nds):
    #print("Loading Price List. Please Wait...")
    with open(plfp, encoding="utf8") as data:
        pl = [[0 for x in range(nds)] for y in range(ns)] # pl: price_list
        sni = 0 # sni:stock_name_index
        for l in data: # l: line
            l = re.sub(" ", "_", str(l))
            for dsi in range(0, nds): # dsi: data_set_index
                pl[sni][dsi] = l.split()[dsi]
            sni += 1
    #print("Done!")
    return pl

def NORMALIZE_FEATURES(X, ntds, ntf):
    nX = np.zeros(shape=(ntds, ntf)) # nX: normalized_X
    mean = [0 for c in range(0, ntf)]
    std = [0 for c in range(0, ntf)]
    nX[:, 0] = X[:, 0]
    # normalizing X using STD and MEAN
    for c in range(1, ntf): # c: column
        mean[c] = np.mean(X, axis=0)[c]
        std[c] = np.std(X, axis=0, ddof=1)[c]
        if std[c] == 0:
            std[c] = 1
        nX[:, c] = (X[:, c] - mean[c]) / std[c]
    return nX, mean, std







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


def LEARN_BY_NELR(si, pl, ntds, ntf, nds, pefl, ntd, reg, demo): # NELR: NORMAL_EQUATION_based_LINEAR_REGRESSION
    if demo == "ON":
        print("\n" + "Linear Regression is ON!")
    # building training data
    rm = np.zeros((ntf, ntf), float) # rm: regularization_matrix
    np.fill_diagonal(rm, reg)
    rm[0][0] = 0
    tX = [[0 for c in range(0, ntf)] for row in range(0, ntds)] # tX: training_X
    tY = [0 for r in range(0, ntds)] # tY: training_Y
    for r in range(0, ntds): # r: row
        tX[r][0] = 1
        for c in range(1, ntf): # c: column
            tX[r][c] = int(float(pl[si][r + c - 1]))
    for r in range(0, ntds): # r: row
        tY[r] = int(float(pl[si][r + ntf - 1]))
    BUILD_OCTAVE_TEST_FILE(tX, tY, ntds, ntf)
    tX = np.array(tX)
    tY = np.array(tY)
    ntX, mean, std = NORMALIZE_FEATURES(tX, ntds, ntf) # ntX: normalized_training_X
    # calculating learning coefficients
    temp1 = np.matmul(ntX.transpose(), ntX)
    temp2 = np.matmul(np.linalg.pinv(temp1 + rm), ntX.transpose())
    theta = np.matmul(temp2, tY)
    # generating predicted Y for training data
    tpY = np.matmul(ntX, theta) # tpY: training_predicted_Y
    # demonstrating training phase
    if demo == "ON":
        print("MSE on training set: " + str(MSE(tpY, tY)))
        print("MAE on training set: " + str(MAE(tpY, tY)))
        PLOT_DIAGRAM(pefl, tY, tpY, 0, 0)












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