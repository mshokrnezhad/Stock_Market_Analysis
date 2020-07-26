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
    # print("Loading Stock List. Please Wait...")
    # reading processed file
    with open(slfp, encoding="utf8") as data:
        sl = [] # sl: stock_list
        for l in data: # l: line
            l = re.sub(" ", "_", str(l))
            sl.append(str(l.split()[0]))
    # print("Done!")
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
    # print("Loading Price List. Please Wait...")
    with open(plfp, encoding="utf8") as data:
        pl = [[0 for x in range(nds)] for y in range(ns)] # pl: price_list
        sni = 0 # sni:stock_name_index
        for l in data: # l: line
            l = re.sub(" ", "_", str(l))
            for dsi in range(0, nds): # dsi: data_set_index
                pl[sni][dsi] = l.split()[dsi]
            sni += 1
    # print("Done!")
    return pl

def NORMALIZE_FEATURES(X, ntds, ntf):
    nX = np.zeros(shape=(ntds, ntf)) # nX: normalized_X
    mean = [0 for x in range(0, ntf)]
    std = [0 for x in range(0, ntf)]
    nX[:, 0] = X[:, 0]
    # normalizing X using STD and MEAN
    for c in range(1, ntf): # c: column
        mean[c] = np.mean(X, axis=0)[c]
        std[c] = np.std(X, axis=0, ddof=1)[c]
        if std[c] == 0:
            std[c] = 1
        nX[:, c] = (X[:, c] - mean[c]) / std[c]
    return nX, mean, std

def BUILD_OCTAVE_TEST_FILE(X, Y, ntds, ntf):
    fn = "text_files/test.txt" # fn: file_name
    open("text_files/test.txt", 'w').close()
    # generating lines
    nl = [""] * ntds # nl: new_line
    for r in range(0, ntds): # r: row
        for c in range(1, ntf): # c: column
            nl[r] = nl[r] + str(X[r][c]) + ","
        nl[r] = nl[r] + str(Y[r])
    # writing file
    f = open(fn, "w") # f: file
    for r in range(0, ntds):
        f.write(nl[r])
        if r < ntds - 1:
            f.write("\n")
    f.close()

def MSE(pY, Y): # mean squared error, pY: predicted_Y
    return np.dot((pY - Y), (pY - Y)) / (2 * len(Y))

def MAE(pY, Y): # mean absolute error, pY: predicted_Y
    return np.sum(np.abs(pY - Y))/(2 * len(Y))

def PROCESS_EXCEL_FILES_LIST(efp, nds):
    efl = [f for f in listdir(efp) if isfile(join(efp, f))]
    l = np.arange(0, nds, 1) # l: label
    pefl = [""] * len(l)
    # generating lists
    for fni in range(0, len(l)): # fni: file_name_index
        pefl[fni] = efl[l[fni]]
    for fn in range(0, len(pefl)): # fn: file_name
        pefl[fn] = re.sub('.xlsx', "", re.sub('_', "", re.sub('MarketWatchPlus-13', "", pefl[fn])))
    return efl, pefl

def PLOT_TRAINING_DIAGRAM(pefl, Y, pY, li, ntd): # pY: predicted_Y, li: label_index
    #  generating x axis label
    if ntd > 0:
        ed = ["" for x in range(ntd)] # ed: extra_days
        for d in range(ntd): # d: day
            ed[d] = "day+" + str(d + 1)
        pefl =  pefl + ed
    pefl = np.array(pefl)
    tefl =np.split(pefl, [li, len(Y) + li])[1] # tefl: temporary_efl
    # plotting diagram
    l = np.arange(li, li + len(Y), 1) # l: label
    plt.plot(l, Y, 'r--', l, pY, 'b*')
    plt.xticks(l, tefl, rotation='vertical')
    plt.grid()
    plt.show()

def PLOT_PREDICTION_DIAGRAM(pnd, p): # pnd: price_of_next_days, p: price
    # generating x axis label for prediction phase
    Xl = np.arange(0, len(pnd), 1) # Xl: X_label
    lpl = [int(float(p)) for x in range(len(pnd))] # lpl: last_price_line
    ed = ["" for x in range(len(pnd))] # ed: extra_days
    for d in range(0, len(pnd)): # d: day
        ed[d] = "day+" + str(d + 1)
    # plotting diagram
    plt.plot(Xl, lpl, 'r--', pnd, 'b*')
    plt.xticks(Xl, ed, rotation='vertical')
    ax = plt.gca()
    ax.grid(axis='both', which='both')
    plt.show()

def CORRECT_LENGTH(s, l): # add spaces at the end of s to make its l equal to the given l, s: string, l: length
    if len(s) < l:
        for i in range(l-len(s)):
            s = s + " "
    return s

def LEARN_BY_NELR(si, pl, ntds, ntf, pefl, reg, demo): # NELR: NORMAL_EQUATION_based_LINEAR_REGRESSION
    if demo == "ON":
        print("\n" + "Linear Regression is ON!")
    # building training data
    rm = np.zeros((ntf, ntf), float) # rm: regularization_matrix
    np.fill_diagonal(rm, reg)
    rm[0][0] = 0
    tX = [[0 for x in range(0, ntf)] for y in range(0, ntds)] # tX: training_X
    tY = [0 for x in range(0, ntds)] # tY: training_Y
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
        PLOT_TRAINING_DIAGRAM(pefl, tY, tpY, 0, 0)
    return mean, std, theta

def VALIDATE_LRNE_DBD(pefl, nds, ntds, ntf, pl, si, mean, std, theta, demo):  # DBD: day by day
    vY = [0 for x in range(0, nds - (ntds + ntf - 1))] # vY: validation_Y
    vX = [[0 for x in range(0, ntf)] for y in range(0, nds - (ntds + ntf - 1))] # vX: validation_X
    nvX = np.zeros(shape=(nds - (ntds + ntf - 1), ntf)) # nvX: normalized_validation_X
    # building validation data
    for r in range(0, len(vY)): # r: row
        if (ntds + ntf - 1 + r) < nds:
            vY[r] = int(float(pl[si][ntds + ntf - 1 + r]))
    for r in range(0, len(vX)):
        vX[r][0] = 1
        for c in range(1, ntf): # c: col
            vX[r][c] = int(float(pl[si][r + ntds + ntf - 1 - ntf + c]))
    vX = np.array(vX)
    vY = np.array(vY)
    # normalizing data
    nvX[:, 0] = vX[:, 0]
    for c in range(1, ntf):
        nvX[:, c] = (vX[:, c] - mean[c]) / std[c]
    # generating predicted Y for validation data
    vpY = np.matmul(nvX, theta) # vpY: validation_predicted_Y
    # demonstrating validation phase
    if demo == "ON":
        print("MSE on validation set: " + str(MSE(vpY, vY)))
        print("MAE on validation set: " + str(MAE(vpY, vY)))
        PLOT_TRAINING_DIAGRAM(pefl, vY, vpY, ntds + ntf - 1, 0)

# def VALIDATE_LRNE_LT(pefl, nds, ntds, ntf, pl, si, mean, std, theta, demo):  # LT: long term
#     validation_X = [1 for col in range(0, ntf)]
#     normal_validation_X = [1 for col in range(0, ntf)]
#     price_next_day = [0 for row in range(0, nds - ntds)]
#     validation_Y = [0 for row in range(0, nds - ntds)]
#     # building validation data
#     for row in range(0, len(validation_Y)):
#         validation_Y[row] = int(float(pl[stock_id][ntds + 1 + row]))
#     for col in range(1, ntf):
#         validation_X[col] = int(float(pl[stock_id][ntds - ntf + col]))
#     # calculating next day prices
#     for day in range(0, len(price_next_day)):
#         for col in range(1, ntf):
#             normal_validation_X[col] = (validation_X[col] - mean[col]) / std[col]
#         price_next_day[day] = math.ceil(np.matmul(normal_validation_X, theta))
#         validation_X = np.roll(validation_X, -1)
#         validation_X[ntf - 1] = price_next_day[day]
#         validation_X[0] = 1
#
#     if demonstration == "ON":
#         print("Expected price of next day: " + str(price_next_day[0]))
#         prediction_diagram(price_next_day, price_list[stock_id][number_of_dataset_rows - 1])

def PREDICT_LRNE(ntd, ntf, nds, pl, si, mean, std, theta, demo):
    ndp = [0 for x in range(0, ntd)] # ndp: next_days_prices
    ndX = [1 for x in range(0, ntf)] # ndX: next_days_X
    nndX = [1 for x in range(0, ntf)] # nndx: normalized_next_days_X
    for c in range(1, ntf): # c: column
        ndX[c] = int(float(pl[si][nds - ntf + c]))
    # calculating netx days prices
    for d in range(0, ntd): # d: day
        for c in range(1, ntf):
            nndX[c] = (ndX[c] - mean[c]) / std[c]
        ndp[d] = math.ceil(np.matmul(nndX, theta))
        ndX = np.roll(ndX, -1)
        ndX[ntf-1] = ndp[d]
        ndX[0] = 1
    # demonstrating prediction results
    if demo == "ON":
        print("Expected price of next day: " + str(ndp[0]))
        PLOT_PREDICTION_DIAGRAM(ndp, pl[si][nds-1])
    return ndp, pl[si][nds-1]




