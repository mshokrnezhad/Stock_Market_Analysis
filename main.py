from functions import *

excel_files_path = "excel_files"
stock_list_file_path = "text_files/stock_list.txt"
price_list_file_path = "text_files/price_list.txt"

# stock_list = get_stock_list_from_excel_files(excel_files_path, stock_list_file_path)
stock_list = get_stock_list_from_processed_file(stock_list_file_path)

# price_list = get_price_list_from_excel_files(stock_list, excel_files_path, price_list_file_path)
price_list = get_price_list_from_processed_file(price_list_file_path, len(stock_list), 68)

# print(price_list)

# we have 68 columns of data.
# we decide to use the price of past 10 days to predict the price of day 11,
# so we have X = [1 P_d_-10 P_d_-9 ... P_d_-1] and y = p_d_0.
# also we decide to use 40 sets of data to train h function.
# lets go

alpha = 1
number_of_training_sets = 40
number_of_training_features = 2 #11
number_of_iterations = 1000
X = [[0 for col in range(0, number_of_training_features)] for row in range(0, number_of_training_sets)]
Y = [0 for row in range(0, number_of_training_sets)]
theta = [0 for col in range(0, number_of_training_features)]
error = 0

for row in range(0, number_of_training_sets):
    X[row][0] = 1
    for col in range(1, number_of_training_features):
        X[row][col] = int(float(price_list[1][row+col])) # 1->0

for row in range(0, number_of_training_sets):
    Y[row] = int(float(price_list[1][row+number_of_training_features]))  # 1->0


file_name = "text_files/test.txt"
open("text_files/test.txt", 'w').close()

line_index = 0
new_line = [""] * number_of_training_sets
for row in range(0, number_of_training_sets):
    for col in range(1, number_of_training_features):
        new_line[row] = new_line[row] + str(X[row][col]) + ","
    for col in range(1, number_of_training_features):
        new_line[row] = new_line[row] + str(X[row][col]*X[row][col]) + ","
    for col in range(1, number_of_training_features):
        new_line[row] = new_line[row] + str(X[row][col]*X[row][col]*X[row][col]) + ","
    new_line[row] = new_line[row] + str(Y[row])

file = open(file_name, "w")
for n in range(0, number_of_training_sets):
    file.write(new_line[n])
    if n < number_of_training_sets - 1:
        file.write("\n")
file.close()



X_times_theta = [0 for row in range(0, number_of_training_sets)]
X_times_theta_minus_Y = [0 for row in range(0, number_of_training_sets)]
col_sum_X = [0 for col in range(0, number_of_training_features)]
processed_X = [[0 for col in range(0, number_of_training_features)] for row in range(0, number_of_training_sets)]

for itr in range(0, 1000):

    for row in range(0, number_of_training_sets):
        for col in range(0, number_of_training_features):
            X_times_theta[row] = X_times_theta[row] + theta[col]*X[row][col]
        X_times_theta_minus_Y[row] = X_times_theta[row] - Y[row]
        for col in range(0, number_of_training_features):
            processed_X[row][col] = X[row][col] * X_times_theta_minus_Y[row]

    for col in range(0, number_of_training_features):
        for row in range(0, number_of_training_sets):
            col_sum_X[col] = col_sum_X[col] + processed_X[row][col]
        theta[col] = theta[col] - (alpha * col_sum_X[col]) / number_of_training_sets


for row in range(0, number_of_training_sets):
    for col in range(0, number_of_training_features):
        X_times_theta[row] = X_times_theta[row] + theta[col]*X[row][col]
    print(str(X_times_theta[row]) + " for " + str(Y[row]))
    X_times_theta_minus_Y[row] = X_times_theta[row] - Y[row]
    X_times_theta_minus_Y[row] = X_times_theta_minus_Y[row] * X_times_theta_minus_Y[row]
    error = error + X_times_theta_minus_Y[row]

error = 1/(2 * number_of_training_sets * error)



print(str(error))


# for iter = 1:num_iters

# 	theta = theta - alpha*(1/m)*(sum(((X*theta)-y).*X))';
