from functions import *

excel_files_path = "excel_files"
stock_list_file_path = "text_files/stock_list.txt"
price_list_file_path = "text_files/price_list.txt"

# stock_list = get_stock_list_from_excel_files(excel_files_path, stock_list_file_path)
stock_list = get_stock_list_from_processed_file(stock_list_file_path)

# price_list = get_price_list_from_excel_files(stock_list, excel_files_path, price_list_file_path)
price_list = get_price_list_from_processed_file(price_list_file_path, len(stock_list), 68)

print(price_list)