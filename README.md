# Stock_Market_Analysis
To Analyse the Iran's Stock Market using Normal Equation based Linear Regression 
In this code, for each stock, the price of each day is predicted based on the previous prices of that stock at the previous days.
To run this code, the market history is retrieved form http://www.tsetmc.com/Loader.aspx?ParTree=15131F# (see files in the "excel files" folder)

The main file has three parts:

# 1) Analysing a specific stock
- Getting the name of a stock 
- Calculating learning coefficients 
- Next, there is a day by day validation phase. In this phase, the predicted price of each day of the validation period is calculated based on the set of previous prices, comes from the validation dataset.
- Then, there is also a long term validation phase. In this phase, the price of Day 1 of the validation period is calculated based on the set of previous prices, comes from the validation dataset. Then the predicted price of Day 1 is added to the set of previous prices and used to predict the price of Day 2, and so on. In othe words, in this validation phase, unlike the day by day validation, the currently predicted data is also considered in the prediction procedure of future prices.
- Finally, the growth percentage of the stock is demonstrated for next 1 day, 1 week, 1 month, 2 months, and 3 months.

# 2) Analysing a list of stocks
- Getting the list of stock names as a text file
- Simply running the procedure defined in 1 (Analysing a specific stock) in a loop

# 3) Finding the most profitable stocks
- Simply running the procedure defined in 1 (Analysing a specific stock) in a loop for all available stocks in the market
- Calculating the most profitable stocks for next 1 day, 1 week, 1 month, 2 months, and 3 months based on their market growth 

You can uncomment each part you want and simply run the main file.
Enjoy it!
