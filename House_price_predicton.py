#  Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("housing_data.csv")

#  Define the feature matrix (X) and the target vector (y)
X = df[['House_Size', 'Number_of_Bedrooms']]  # Multiple features
y = df['Price']  # Target variable

#  Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

#  Make predictions on the test data
y_pred = model.predict(X_test)

#  Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# # Print the weights (coefficients) and intercept of the model
weights = model.coef_
intercept = model.intercept_

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")



#  Predict the price for a new house with multiple features
new_house = pd.DataFrame([[1300, 3]], columns=['House_Size', 'Number_of_Bedrooms'])
predicted_price_new_house = model.predict(new_house)


print(f"The predicted price for a house with size 1300 sq ft, 3 bedrooms, and 7 years old is: ${predicted_price_new_house[0]:.2f}")


#  Predict the price for a new house size (e.g., 340 sq ft)
new_house_size = pd.DataFrame([[1200,3]],columns=['House_Size', 'Number_of_Bedrooms'])  # New house size (in square feet)
predicted_price = model.predict(new_house_size)

print(f"The predicted price for a house with size 340 sq ft is: ${predicted_price[0]:.2f}")

