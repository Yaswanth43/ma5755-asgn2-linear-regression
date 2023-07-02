import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# Load the synthetic data
data = pd.read_csv('datafile.txt')
pd.set_option('display.expand_frame_repr', False)
print(data.describe())

# Taking a sample of 150 out of the population
sample_size = 150

(data.sample(sample_size))

spdata = data.sample(sample_size)

# Select the relevant features and target variable
X = spdata[['Date','Open', 'High', 'Low', 'Close' , 'Volume']]

# I have used adj close as stock price, the response variable in this case
Y = spdata['Adj Close']

# Convert categorical features to numerical
X = pd.get_dummies(X, columns=['Date'])

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

# Train a linear regression model 
model = LinearRegression()
model.fit(X_train, Y_train)

# Predict the stock price for the test set
Y_pred = model.predict(X_test)

# Calculate the accuracy (R^2 score)
accuracy = model.score(X_test, Y_test)

# Evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error
print("Mean Absolute Error: ", mean_absolute_error(Y_test, Y_pred))
print("Mean Squared Error: ", mean_squared_error(Y_test, Y_pred))

#defining each attribute in the sample data
O = spdata[['Open']]
C = spdata[['Close']]
H = spdata[['High']]
L = spdata[['Low']]
V = spdata[['Volume']]

# Split the data of each attribute into training and testing sets
O_train, O_test, Y_train, Y_test = train_test_split(O, Y, test_size=0.5)
C_train, C_test, Y_train, Y_test = train_test_split(C, Y, test_size=0.5)
H_train, H_test, Y_train, Y_test = train_test_split(H, Y, test_size=0.5)
L_train, L_test, Y_train, Y_test = train_test_split(L, Y, test_size=0.5)
V_train, V_test, Y_train, Y_test = train_test_split(V, Y, test_size=0.5)

# Train a linear regression model and pridicting the stock price as a fuction of opening value
model = LinearRegression()
model.fit(O_train, Y_train)
O_pred = model.predict(O_test)

# Train a linear regression model and pridicting the stock price as a fuction of closing value
model = LinearRegression()
model.fit(C_train, Y_train)
C_pred = model.predict(C_test)

# Train a linear regression model and pridicting the stock price as a fuction of highest value
model = LinearRegression()
model.fit(H_train, Y_train)
H_pred = model.predict(H_test)

# Train a linear regression model and predicting the stock price as a fuction of lowest value
model = LinearRegression()
model.fit(L_train, Y_train)
L_pred = model.predict(L_test)

# Train a linear regression model and pridicting the stock price as a fuction of volume
model = LinearRegression()
model.fit(V_train, Y_train)
V_pred = model.predict(V_test)

#evaluating each model
print("\nMean Absolute Error of open: ", mean_absolute_error(Y_test, O_pred))
print("Mean Squared Error of open: ", mean_squared_error(Y_test, O_pred))

print("\nMean Absolute Error of close: ", mean_absolute_error(Y_test, C_pred))
print("Mean Squared Error of close: ", mean_squared_error(Y_test, C_pred))

print("\nMean Absolute Error of high: ", mean_absolute_error(Y_test, H_pred))
print("Mean Squared Error of high: ", mean_squared_error(Y_test, H_pred))

print("\nMean Absolute Error of low: ", mean_absolute_error(Y_test, L_pred))
print("Mean Squared Error of low: ", mean_squared_error(Y_test, L_pred))

print("\nMean Absolute Error of volume: ", mean_absolute_error(Y_test, V_pred))
print("Mean Squared Error of volume: ", mean_squared_error(Y_test, V_pred))

# selecting a small sample size to clearly view each datapoint in scatterplot matrix
sample_size = 100

(data.sample(sample_size))

spldata = data.sample(sample_size)

# Select the numerical columns
spldata = spldata[['Adj Close','Open', 'High', 'Low', 'Close','Volume' ]]

# Create the scatterplot matrix
scatter_matrix(spldata, alpha=0.5, figsize=(6, 6), diagonal='none')
plt.show()

# getting data from low, by inspecting scatterplot and selecting the best fit line, I have taken only low because close data 'as the internet says' can also represent the stock price
M = data[['Low']]
A = data[['Adj Close']]

M_train, M_test, A_train, A_test = train_test_split(M, A, test_size=0.5)

linreg = LinearRegression()
linreg.fit(M_train,A_train)
M_pred=linreg.predict(M_test)

# Evaluate
print("\nMean Absolute Error: ", mean_absolute_error(M_pred, A_test))
print("Mean Squared Error: ", mean_squared_error(M_pred, A_test))
print("Accuracy: ", accuracy)
