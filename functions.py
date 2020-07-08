import re
import xlrd
from os import listdir
from os.path import isfile, join


def get_stock_list_from_excel_files(excel_files_path, processed_file_path):

    print("Extracting Stock List from Excel Files. Please Wait...")

    excel_files_list = [f for f in listdir(excel_files_path) if isfile(join(excel_files_path, f))]
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

    print("Loading Stock List. Please Wait...")

    with open(processed_file_path, encoding="utf8") as data:
        stock_list = []
        for line in data:
            line = re.sub(" ", "_", str(line))
            part = line.split()
            stock_list.append(str(part[0]))

    print("Done!")

    return stock_list


def build_preliminary_file_of_zeros(file_name, number_of_stocks):
    prices_file = open(file_name, "w")
    for n in range(0, number_of_stocks):
        prices_file.write("1")
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

    return wb_stock_list,  wb_prices


def build_prices_file(file_name, name, price):
    line_index = 0
    new_line = [""] * len(name)
    with open(file_name) as data:
        for line in data:
            line = line.rstrip("\n")
            new_line[line_index] = str(line) + "\t" + str(price[line_index])
            line_index += 1

    open(file_name, 'w').close()

    prices_file = open(file_name, "w")
    for n in range(0, len(name)):
        prices_file.write(new_line[n])
        if n < len(name) - 1:
            prices_file.write("\n")
    prices_file.close()
