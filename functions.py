import re
import xlrd
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt


def get_stock_list_from_excel_files(excel_files_path, processed_file_path, excel_files_list):
    print("Extracting Stock List from Excel Files. Please Wait...")

    stock_list = ["وتجارت"]

    for excel_file in excel_files_list:
        excel_sheet = xlrd.open_workbook(excel_files_path + "/" + excel_file).sheet_by_index(0)
        work_book = [""] * (excel_sheet.nrows - 3)
        for r in range(3, excel_sheet.nrows):
            work_book[r - 3] = str(excel_sheet.cell_value(r, 0))
        for sn in work_book:
            if sn not in stock_list:
                stock_list.append(sn)

    open(processed_file_path, 'w').close()
    stock_list_file = open(processed_file_path, "w", encoding="utf8")
    for n in range(0, len(stock_list)):
        stock_list_file.write(stock_list[n])
        if n < len(stock_list) - 1:
            stock_list_file.write("\n")
    stock_list_file.close()

    print("Done!")

    return stock_list


def get_stock_list_from_processed_file(processed_file_path):
    #print("Loading Stock List. Please Wait...")

    with open(processed_file_path, encoding="utf8") as data:
        stock_list = []
        for line in data:
            line = re.sub(" ", "_", str(line))
            part = line.split()
            stock_list.append(str(part[0]))

    #print("Done!")

    return stock_list


def build_preliminary_file(file_name, number_of_stocks):
    prices_file = open(file_name, "w")
    for n in range(0, number_of_stocks):
        prices_file.write("xxx")
        if n < number_of_stocks - 1:
            prices_file.write("\n")
    prices_file.close()


def get_work_book_info(file_name):
    excel_file = xlrd.open_workbook(file_name)
    sheet = excel_file.sheet_by_index(0)
    wb_stock_list = [""] * (sheet.nrows - 3)
    wb_prices = [0] * (sheet.nrows - 3)

    for r in range(3, sheet.nrows):
        wb_stock_list[r - 3] = re.sub(" ", "_", str(sheet.cell_value(r, 0)))
        wb_prices[r - 3] = sheet.cell_value(r, 10)

    return wb_stock_list, wb_prices


def build_prices_file(file_name, name, price):
    line_index = 0
    new_line = [""] * len(name)
    with open(file_name) as data:
        for line in data:
            line = line.rstrip("\n")
            if line == "xxx":
                new_line[line_index] = str(price[line_index])
            else:
                new_line[line_index] = str(line) + "\t" + str(price[line_index])
            line_index += 1

    open(file_name, 'w').close()

    prices_file = open(file_name, "w")
    for n in range(0, len(name)):
        prices_file.write(new_line[n])
        if n < len(name) - 1:
            prices_file.write("\n")
    prices_file.close()


def get_price_list_from_excel_files(stock_list, excel_files_path, price_list_file_path):
    build_preliminary_file(price_list_file_path, len(stock_list))
    excel_files_list = [f for f in listdir(excel_files_path) if isfile(join(excel_files_path, f))]

    print("Generating Prices File. Please Wait...")

    price_list = [[0 for x in range(len(excel_files_list))] for y in range(len(stock_list))]
    last_valid_price = [0 for x in range(len(stock_list))]

    excel_file_index = 0
    for excel_file in excel_files_list:
        current_price = [0] * len(stock_list)
        (wb_stocks, wb_prices) = get_work_book_info(excel_files_path + "/" + excel_file)

        for s in stock_list:
            if s in wb_stocks:
                if wb_prices[wb_stocks.index(s)] != 0:
                    current_price[stock_list.index(s)] = wb_prices[wb_stocks.index(s)]
                    last_valid_price[stock_list.index(s)] = wb_prices[wb_stocks.index(s)]
                elif wb_prices[wb_stocks.index(s)] == 0 and last_valid_price[stock_list.index(s)] != 0:
                    current_price[stock_list.index(s)] = last_valid_price[stock_list.index(s)]
                elif wb_prices[wb_stocks.index(s)] == 0 and last_valid_price[stock_list.index(s)] == 0:
                    current_price[stock_list.index(s)] = 0
            else:
                current_price[stock_list.index(s)] = last_valid_price[stock_list.index(s)]

        temp_stock_index = 0
        for pr in price_list:
            pr[excel_file_index] = current_price[temp_stock_index]
            temp_stock_index += 1
        excel_file_index += 1
        build_prices_file(price_list_file_path, stock_list, current_price)

    print("Done!")

    return price_list


def get_price_list_from_processed_file(processed_file_path, number_of_rows, number_of_columns):
    #print("Loading Price List. Please Wait...")

    with open(processed_file_path, encoding="utf8") as data:
        price_list = [[0 for x in range(number_of_columns)] for y in range(number_of_rows)]
        row_index = 0
        for line in data:
            line = re.sub(" ", "_", str(line))
            part = line.split()
            for col_index in range(0, number_of_columns):
                price_list[row_index][col_index] = part[col_index]
            row_index += 1

    #print("Done!")

    return price_list


def normalize_features(X, number_of_rows, number_of_columns):
    Z = np.zeros(shape=(number_of_rows, number_of_columns))
    mean = [0 for m in range(0, number_of_columns)]
    std = [0 for m in range(0, number_of_columns)]

    Z[:, 0] = X[:, 0]
    for col in range(1, number_of_columns):
        mean[col] = np.mean(X, axis=0)[col]
        std[col] = np.std(X, axis=0, ddof=1)[col]
        Z[:, col] = (X[:, col] - mean[col]) / std[col]

    return Z, mean, std


def linear_regression_NE(stock_id, price_list, number_of_training_rows, number_of_feature_columns,
                         number_of_dataset_rows, processed_excel_files_list, prediction_day):
    # number_of_training_rows + prediction_day <= number_of_dataset_rows

    print("\n" + "Linear Regression is ON!")

    training_X = [[0 for col in range(0, number_of_feature_columns)] for row in range(0, number_of_training_rows)]
    training_Y = [0 for row in range(0, number_of_training_rows)]
    for row in range(0, number_of_training_rows):
        training_X[row][0] = 1
        for col in range(1, number_of_feature_columns):
            training_X[row][col] = int(float(price_list[stock_id][row + col - 1]))
    for row in range(0, number_of_training_rows):
        training_Y[row] = int(float(price_list[stock_id][row + number_of_feature_columns - 1 + prediction_day]))

    build_octave_test_text_file(training_X, training_Y, number_of_training_rows, number_of_feature_columns)

    training_X = np.array(training_X)
    training_Y = np.array(training_Y)

    normal_training_X, mean, std = normalize_features(training_X, number_of_training_rows, number_of_feature_columns)

    theta = np.matmul(np.matmul(np.linalg.pinv(np.matmul(normal_training_X.transpose(), normal_training_X)),
                                normal_training_X.transpose()), training_Y)

    training_predicted_Y = np.matmul(normal_training_X, theta)

    print("MSE on training set: " + str(mean_squared_error(training_predicted_Y, training_Y)))
    print("MAE on training set: " + str(mean_absolute_error(training_predicted_Y, training_Y)))
    diagram_plot(processed_excel_files_list, training_Y, training_predicted_Y, 0, prediction_day)

    validation_Y = [0 for row in range(0, number_of_dataset_rows - (number_of_training_rows + number_of_feature_columns - 1))]
    validation_X = [[0 for col in range(0, number_of_feature_columns)] for row in
                    range(0, number_of_dataset_rows - (number_of_training_rows + number_of_feature_columns - 1))]
    normal_validation_X = np.zeros(shape=(number_of_dataset_rows - (number_of_training_rows + number_of_feature_columns - 1),
                                          number_of_feature_columns))
    X_next_day = [1 for col in range(0, number_of_feature_columns)]
    normal_X_next_day = [1 for col in range(0, number_of_feature_columns)]

    for row in range(0, len(validation_Y)):
        if (number_of_training_rows + number_of_feature_columns - 1 + row + prediction_day) < number_of_dataset_rows:
            validation_Y[row] = int(float(price_list[stock_id][number_of_training_rows + number_of_feature_columns - 1
                                                               + row + prediction_day]))
    for row in range(0, len(validation_X)):
        validation_X[row][0] = 1
        for col in range(1, number_of_feature_columns):
            validation_X[row][col] = int(float(price_list[stock_id][row + number_of_training_rows + number_of_feature_columns
                                                                    - 1 - number_of_feature_columns + col]))
    for col in range(1, number_of_feature_columns):
        X_next_day[col] = int(float(price_list[stock_id][number_of_dataset_rows - number_of_feature_columns + col]))
        normal_X_next_day[col] = (X_next_day[col] - mean[col]) / std[col]

    validation_X = np.array(validation_X)
    validation_Y = np.array(validation_Y)

    normal_validation_X[:, 0] = validation_X[:, 0]
    for col in range(1, number_of_feature_columns):
        normal_validation_X[:, col] = (validation_X[:, col] - mean[col]) / std[col]

    validation_predicted_Y = np.matmul(normal_validation_X, theta)

    print("MSE on validation set: " + str(mean_squared_error(validation_predicted_Y, validation_Y)))
    print("MAE on validation set: " + str(mean_absolute_error(validation_predicted_Y, validation_Y)))

    diagram_plot(processed_excel_files_list, validation_Y, validation_predicted_Y, number_of_training_rows +
                 number_of_feature_columns - 1, prediction_day)

    prediction_next_day = np.matmul(normal_X_next_day, theta)
    print("Expected price of next day: " + str(prediction_next_day))


def build_octave_test_text_file(X, Y, number_of_rows, number_of_columns):
    file_name = "text_files/test.txt"
    open("text_files/test.txt", 'w').close()
    line_index = 0
    new_line = [""] * number_of_rows
    for row in range(0, number_of_rows):
        for col in range(1, number_of_columns):
            new_line[row] = new_line[row] + str(X[row][col]) + ","
        new_line[row] = new_line[row] + str(Y[row])

    file = open(file_name, "w")
    for n in range(0, number_of_rows):
        file.write(new_line[n])
        if n < number_of_rows - 1:
            file.write("\n")
    file.close()


def mean_squared_error(predicted_Y, Y):
    return np.dot((predicted_Y - Y), (predicted_Y - Y))\
           /(2 * len(Y))


def mean_absolute_error(predicted_Y, Y):
    return np.sum(np.abs(predicted_Y - Y))/(2 * len(Y))


def excel_files_list_processor(excel_files_path, number_of_data_sets):
    excel_files_list = [f for f in listdir(excel_files_path) if isfile(join(excel_files_path, f))]
    label = np.arange(0, number_of_data_sets, 1)
    processed_excel_files_list = [""] * len(label)

    for fn_index in range(0, len(label)):
        processed_excel_files_list[fn_index] = excel_files_list[label[fn_index]]
    for fn in range(0, len(processed_excel_files_list)):
        processed_excel_files_list[fn] = re.sub('.xlsx', "", re.sub('_', "", re.sub('MarketWatchPlus-13', "",
                                                                                    processed_excel_files_list[fn])))

    return excel_files_list, processed_excel_files_list


def diagram_plot(processed_excel_files_list, Y, predicted_Y, label_index, prediction_day):

    if prediction_day > 0:
        extra_days = ["" for ed in range(prediction_day)]
        for ed in range(prediction_day):
            extra_days[ed] = "day+" + str(ed+1)
        processed_excel_files_list =  processed_excel_files_list + extra_days

    processed_excel_files_list = np.array(processed_excel_files_list)
    temp_excel_file_list =np.split(processed_excel_files_list, [label_index + prediction_day,
                                                                len(Y) + label_index + prediction_day])[1]

    label = np.arange(label_index + prediction_day, label_index + prediction_day + len(Y), 1)
    plt.plot(label, Y, 'r--', label, predicted_Y, 'b*')
    plt.xticks(label, temp_excel_file_list, rotation='vertical')
    plt.grid()
    plt.show()