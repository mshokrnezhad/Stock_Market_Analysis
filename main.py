from functions import *

excel_files_path = "excel_files"
stock_list_file_path = "text_files/stock_list.txt"
price_list_file_path = "text_files/price_list.txt"

# stock_list = get_stock_list_from_excel_files(excel_files_path, "text_files/stock_list.txt")
stock_list = get_stock_list_from_processed_file(stock_list_file_path)




build_preliminary_file_of_zeros(price_list_file_path, len(stock_list))
excel_files_list = [f for f in listdir(excel_files_path) if isfile(join(excel_files_path, f))]
print("Generating Prices File. Please Wait...")

price_list = [[0 for x in range(len(excel_files_list)+1)] for y in range(len(stock_list))]
for pr in price_list:
    pr[0] = 1

excel_file_index = 1
for excel_file in excel_files_list:
    current_price = [0]*len(stock_list)
    (wb_stocks, wb_prices) = get_work_book_info(excel_files_path + "/" + excel_file)

    for wbs in wb_stocks:
        current_price[stock_list.index(wbs)] = wb_prices[wb_stocks.index(wbs)]

    temp_stock_index = 0
    for pr in price_list:
        pr[excel_file_index] = current_price[temp_stock_index]
        temp_stock_index += 1
    excel_file_index += 1

    build_prices_file(price_list_file_path, stock_list, current_price)

print("Done!")


print(price_list)