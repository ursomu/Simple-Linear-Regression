# Simple-Linear-Regression
Description: This repository contains a basic implementation of Simple Linear Regression, a fundamental supervised learning algorithm. The project  model training, and evaluation with visualizations to illustrate the linear relationship between variables. Suitable for beginners in machine learning.

What is Linear Regression : 
Linear Regression is a statistical method used to model the relationship between a dependent variable (often called the target or response) and one or more independent variables (called features or predictors). The goal is to find the best-fitting straight line (in the case of simple linear regression) or a hyperplane (in the case of multiple linear regression) that minimizes the difference between the observed and predicted values of the dependent variable.


![Screenshot 2024-10-16 152025](https://github.com/user-attachments/assets/165e0166-3512-447e-8244-43e46febaabb)

![image](https://github.com/user-attachments/assets/c85e9000-87f1-4aff-b83f-51c8bb2f5044)


In any machine learning model dataset's play a crutial role. It's very important to select a related data set to your project.

linear regression comes under <b>supervised learning<b> 
<br>
 Supervised learning is a type of machine learning where the model is trained on a labeled dataset, meaning the input data is paired with the correct output. The model learns to map inputs (features) to the corresponding outputs (labels) by finding patterns in the data. The goal is for the model to generalize well enough to make accurate predictions on new, unseen data.


Libraries we have used in our house prediction model : 
import numpy as np  -- Numpy 
import pandas as pd -- Pandas
import matplotlib.pyplot as plt  -- Matplotlib
from sklearn.model_selection import train_test_split  -- sklearn 
from sklearn.linear_model import LinearRegression     -- sklearn 
from sklearn.metrics import mean_squared_error, r2_score -- sklearn 



Numpy : NumPy (Numerical Python) is a powerful Python library used for working with arrays and performing mathematical operations on large datasets efficiently. It's widely used in data science, machine learning, and scientific computing because of its ability to handle and process numerical data quickly and with ease.

Pandas : Pandas is a powerful Python library primarily used for data manipulation, analysis, and handling structured data such as tabular datasets (like Excel spreadsheets or SQL tables). It provides easy-to-use data structures and tools for working with large datasets in a fast and flexible way.

sklern : Scikit-learn (often abbreviated as sklearn) is a widely used Python library for machine learning. It provides simple and efficient tools for data mining and data analysis, built on top of other popular scientific libraries like NumPy, SciPy, and Matplotlib. Scikit-learn is designed to be user-friendly and is suitable for both beginners and experienced practitioners.


House price prediction model : 


~~~

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

# plot the best fit line : 
plt.scatter(X_test.values[:,1], y_test, color='black', label='Actual') 
plt.plot(X_test.values[:,1], y_pred, color='blue', label='Predicted')
plt.xlabel('House size')
plt.ylabel('House Price')
plt.title('House size Vs House price')
plt.legend()
plt.show()

#  Predict the price for a new house with multiple features
new_house = np.array([[1300, 3]])  # Example: 1300 sq ft, 3 bedrooms
predicted_price_new_house = model.predict(new_house)

print(f"The predicted price for a house with size 1300 sq ft, 3 bedrooms, and 7 years old is: ${predicted_price_new_house[0]:.2f}")


#  Predict the price for a new house size (e.g., 340 sq ft, 3 bed rooms)
new_house_size = np.array([[1200,3]])  # New house size (in square feet)
predicted_price = model.predict(new_house_size)

print(f"The predicted price for a house with size 340 sq ft is: ${predicted_price[0]:.2f}")



~~~

Explanatoin of the code : 

We start by importing the necessary libraries:

NumPy: For numerical operations.
Pandas: For data manipulation and analysis.
Matplotlib: For data visualization.
Scikit-learn: For machine learning functionalities, including splitting the dataset, fitting the linear regression model, and evaluating the model.

We load the dataset from a CSV file named housing_data.csv into a Pandas DataFrame called df. This dataset contains information about houses, including their sizes, number of bedrooms, and prices.

Test and Training Data: 

 # X = df[['House_Size', 'Number_of_Bedrooms']]  # Multiple features
 # y = df['Price']  # Target variable
X: This is the feature matrix, which includes the columns House_Size and Number_of_Bedrooms. These features will be used to predict the house price.
y: This is the target vector, which consists of the Price column. This is what we aim to predict based on the features.

Split the Data into Training and Testing Sets:

We split the data into training and testing sets using train_test_split():

test_size=0.2: This means 20% of the data will be used for testing, and 80% will be used for training.
random_state=42: Setting a random state ensures reproducibility, meaning the same random split will occur every time the code is run.

Fit the Linear Regression Model:

We create an instance of the LinearRegression class and fit the model to the training data using model.fit(). This step involves training the model to learn the relationship between the features (X_train) and the target variable (y_train).

Make Predictions on the Test Data:

After the model is trained, we use it to make predictions on the test set (X_test) by calling model.predict(). The predictions are stored in y_pred.

Evaluate the Model:
We evaluate the performance of the model using two metrics:

Mean Squared Error (MSE): This metric measures the average squared difference between the actual (y_test) and predicted (y_pred) prices. A lower MSE indicates a better fit.
R-squared (R²): This metric indicates how well the model explains the variability of the target variable. Values closer to 1 suggest a better fit.

weights and intercept:

We retrieve the coefficients (weights) of the model using model.coef_ and the intercept using model.intercept_. These values indicate the influence of each feature on the predicted price.

plot the best fit line .
We create a NumPy array representing a new house with a size of 1300 square feet and 3 bedrooms. We then use the model to predict its price and print the result.

result for new data : 

![image](https://github.com/user-attachments/assets/012221f6-985e-46d6-8e50-2e827549c57f)

House Price Prediction Model Using Python 
Urs somu
have a nice day.
