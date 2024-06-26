# Sales data

The data set (obtained from [Kaggle](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting/data)) contains sales data for a 'superstore'. The data set contains the following columns:

- **Row ID**: Unique identifier for each row. (index)
- **Order ID**: Unique identifier for each order.
- **Order Date**: Date of the order.
- **Ship Date**: Date of shipping.
- **Ship Mode**: Mode of shipping.
- **Customer ID**: Unique identifier for each customer.
- **Customer Name**: Name of the customer.
- **Segment**: Segment of the customer.
- **Country**: Country of the customer.
- **City**: City of the customer.
- **State**: State of the customer.
- **Postal Code**: Postal code of the customer.
- **Region**: Region of the customer.
- **Product ID**: Unique identifier for each product.
- **Category**: Category of the product.
- **Sub-Category**: Sub-category of the product.
- **Product Name**: Name of the product.
- **Sales**: Sales made.


# Analysis

## 0. Data cleaning

First, some exploration will be done. Checking for missing values, duplicates, data types, etc and solve any issues found.

Postal code for Burlington, Vermont is missing. Filled with appropriate value (5401).

## 1. Data exploration

Eploring (and plotting) the data to understand the distribution of sales in function of the different variables.

By revenue:

- Top customers
- Top cities  
- Top states
- Top categories

# Predictions

Some prediction for the next days.