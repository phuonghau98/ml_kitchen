import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

data = pd.read_csv("Advertising.csv")
data = data.drop(['Unnamed: 0'], axis=1)

plt.plot(data['TV'], data['sales'], 'ro')
plt.xlabel('Spense ($)')
plt.ylabel('Sale (unit)')


X = np.array([[n] for n in data['TV']])
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)
# print(Xbar)

regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(Xbar, data['sales'])
result = regr.coef_

print(result)
drawX = [1, 300]
m = result[1]
b = result[0]
plt.plot(drawX, [m*n + b for n in drawX])
plt.show()
