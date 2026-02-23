# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, separate input features (all columns except price) and the target (price), then split into training and testing sets.

2.Train a Linear Regression model on the training data.

3.Perform 5-fold cross-validation to evaluate model stability and compute average R².

4.Predict on the test set, calculate MSE, MAE, R², and plot actual vs predicted prices to assess performance.

## Program:
```
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: AKILA 
RegisterNumber:  212225220008
*/
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
data=pd.read_csv('CarPrice_Assignment.csv')
data.head()
x=data.drop('price',axis=1)
y=data['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
print('Name: AKILA S')
print('Reg.No: 212225220008')
print("\n=== Cross-Validation ===")
cv_scores=cross_val_score(model,x,y,cv=5)
print("Fold R2 scores:",[f"{score:.4f}" for score in cv_scores])
print(f"Average R2:{cv_scores.mean():.4f}")
y_pred=model.predict(x_test)
print("\n=== Test Set Performance ===")
print(f"MSE: {mean_squared_error(y_test,y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):.2f}")
print(f"R2: {r2_score(y_test,y_pred):.4f}")
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()
```

## Output:
<img width="723" height="150" alt="image" src="https://github.com/user-attachments/assets/b1dd2257-e1e7-4f01-aa45-874c6a1b7978" />
<img width="345" height="113" alt="image" src="https://github.com/user-attachments/assets/e8054afe-019f-4aed-9bff-4536f9a39d3f" />
<img width="1033" height="686" alt="image" src="https://github.com/user-attachments/assets/7ca4127f-7e03-47f4-b7af-b4398d42e517" />





## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
