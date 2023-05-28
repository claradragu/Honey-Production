import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")
df.head()

prod_per_year = df.groupby("year").mean().reset_index()  # Group by "year" instead of "totalprod"
X=prod_per_year["year"]
X = X.values.reshape(-1, 1)  # Use the index as X values
y = prod_per_year["totalprod"]

plt.scatter(X, y)

regr = linear_model.LinearRegression()
regr.fit(X, y)
print(regr.coef_[0])

y_predict = regr.predict(X)
plt.plot(X, y_predict)
plt.show()

X_future = np.array(range(2013, 2051))
X_future = X_future.reshape(-1, 1)
future_predict = regr.predict(X_future)

# Plotting future_predict vs X_future on a separate plot
fig, ax = plt.subplots()
ax.plot(X_future, future_predict)
ax.set(xlabel='Year', ylabel='Total Production', title='Future Honey Production Predictions')
plt.show()

