# Stock Prediction Analysis

This project involves analyzing stock market data and predicting stock prices using linear regression. It uses a sample dataset to train and evaluate the models. The project performs the following steps:

1. Load the synthetic stock market data and explore its statistical summary.
2. Take a sample from the population dataset.
3. Select relevant features and the target variable.
4. Convert categorical features to numerical using one-hot encoding.
5. Split the data into training and testing sets.
6. Train multiple linear regression models to predict stock prices based on different attributes (Open, Close, High, Low, Volume).
7. Evaluate each model using mean absolute error and mean squared error metrics.
8. Visualize the scatterplot matrix to understand the relationships between different numerical variables.
9. Predict the stock price based on the 'Low' attribute and evaluate the model's performance.
10. Calculate and print the accuracy (R^2 score) of the linear regression model.

## Results

The analysis yielded the following results:

- Mean Absolute Error and Mean Squared Error for each model: 
  - Open: MAE, MSE
  - Close: MAE, MSE
  - High: MAE, MSE
  - Low: MAE, MSE
  - Volume: MAE, MSE

- Scatterplot Matrix: A visual representation of the relationships between the 'Adj Close', 'Open', 'High', 'Low', 'Close', and 'Volume' attributes.

- Prediction based on 'Low' attribute: The model predicted stock prices based on the 'Low' attribute, and the evaluation metrics (MAE and MSE) were calculated.

- Accuracy: The coefficient of determination (R^2 score) of the linear regression model was calculated, resulting in an accuracy of 0.9994657051141655.

## Conclusion

The analysis demonstrates the application of linear regression in predicting stock prices. The high accuracy obtained suggests a strong fit of the model to the data. However, it's important to consider the limitations of the dataset and the assumptions made in the analysis.

Feel free to explore the code and dataset provided to gain further insights and modify the analysis as needed.
