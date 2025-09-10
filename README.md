# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load dataset (Hours → X, Scores → Y).

2.Split data into training set and test set.

3.Train a Linear Regression model on training data.

4.Predict marks on test data.

5.Evaluate performance (MAE, MSE, RMSE).

6.Plot regression line with training and test data.

7.Use model to predict marks for new study hours.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: joshitha shree bs
RegisterNumber: 212224230107
*/
```
```
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset (Hours vs Scores)
data = {
    "Hours": [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7,
              7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4,
              2.7, 4.8, 3.8, 6.9, 7.8],
    "Scores": [21, 47, 27, 75, 30, 20, 88, 60, 81, 25,
               85, 62, 41, 42, 17, 95, 30, 24, 67, 69,
               30, 54, 35, 76, 86]
}
df = pd.DataFrame(data)

# Features and target
X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

# Train model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict test results
Y_pred = regressor.predict(X_test)
print("Predicted:", Y_pred)
print("Actual:", Y_test)

# Training set plot
plt.scatter(X_train, Y_train, color="orange")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Test set plot
plt.scatter(X_test, Y_test, color="blue")
plt.plot(X, regressor.predict(X), color="green")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Error metrics
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
```
## Output:
<img width="1119" height="93" alt="image" src="https://github.com/user-attachments/assets/587e3b74-3bf3-4fa9-a4c5-ddfca83177ee" />

<img width="907" height="679" alt="image" src="https://github.com/user-attachments/assets/5a554581-423a-4f2d-9fb5-f098c6b07696" />

<img width="897" height="666" alt="image" src="https://github.com/user-attachments/assets/052c9ea6-a37a-41a0-9cdd-bbf7f78f329e" />

<img width="646" height="97" alt="image" src="https://github.com/user-attachments/assets/4ca4cf2a-3416-420a-a8e3-a062f1ece092" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
