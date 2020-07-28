# Stock_Market_Analysis
To Analyse the Iran's Stock Market using Normal Equation based Linear Regression 
In this code, for each stock, the price of each day is predicted based on the previous prices of that stock at the previous days 

The main file has three parts:
# 1) analysing a specific stock
- Getting the name of a stock 
- Calculating learning coefficients 
- Then, there is a day bt day validation phase. In this phase, the predicted price of each day is calculated based on the set of previous prices, comes from the validation dataset
- Next, there is also a long term validation phase. In this phase, the price of Day 1 of the validation period is calculated based on the set of previous prices, comes from the validation dataset. Then the predicted price of Day 1 is added to the previous prices set and used to predict the price of Day 2, and so on. 
- Finally, the growth percentage of the stock is demonstrated for nex 1 day, 1 week, 1 month, 2 months, and 3 months.

