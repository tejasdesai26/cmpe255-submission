import pandas as pd 
import numpy as np 
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

COLUMNS = ['CRIM','ZN','INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv("housing.csv",delim_whitespace = True,header= None,names=COLUMNS)

#print head of the dataframe
#print(df.head())

#featurecol = ['CRIM','ZN','INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
featurecol = ['RM']
x = df[featurecol]
y = df['MEDV']

#split random 20% data as test data, using LR fit the training data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
linereg = LinearRegression()
linereg.fit(x_train,y_train)
#print(linereg.intercept_)
#print(linereg.coef_)

#Calculate root mean square error
mse = mean_squared_error(y_test, linereg.predict(x_test))
print("Root mean square error: ",np.sqrt(mse))

#r2_score value
#print("R Square Coefficient (R-squared score)", linereg.score(x_test, y_test))

# Calculate r2_score another way
pred_y = linereg.predict(x_test)
print("R2 Square value", r2_score(y_test, pred_y) )

plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, pred_y, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()









