# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
import pandas as pd.
### Step2
Read the csv file.
### Step3
 Get the value of X and y variables
### Step4
 Create the linear regression model and fit.
### Step5
 Predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm cube.
## Program:
```
DEVELOPED BY : SATHEESWARI
REG: 212225240141

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

reg = LinearRegression()
reg.fit(X_train, y_train)

print("Coefficients:", reg.coef_)
print("Variance score:", reg.score(X_test, y_test))

plt.style.use('fivethirtyeight')

plt.scatter(reg.predict(X_train),
            reg.predict(X_train) - y_train,
            color="green", s=10, label="Train data")

plt.scatter(reg.predict(X_test),
            reg.predict(X_test) - y_test,
            color="blue", s=10, label="Test data")

plt.hlines(y=0, xmin=0, xmax=5, linewidth=2)

plt.legend(loc="upper right")
plt.title("Residual errors")
plt.show()




```
## Output:

### Insert your output

<img width="1226" height="772" alt="image" src="https://github.com/user-attachments/assets/e8f53ce8-dc23-4563-a76b-c6f7e4e149bc" />


## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
