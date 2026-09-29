import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data=pd.DataFrame({

    "temperature":[20,22,24,26,28,30,32,34,36,38],

    "sales":[120,135,150,165,180,195,210,225,270,285]})

x=data[["temperature"]]

y = data["sales"]

model=LinearRegression()

model.fit(x,y)

print("\n slope(m):",model.coef_[0])

print("intercept(b):",model.intercept_)

temp=float(input("\n Enter temperature:"))

predicted_sales=model.predict([[temp]])

print("\n---------prediction Result---------")

print("temperature:",temp)

print("Predicted sales:", round(predicted_sales[0], 2), "units")
y_pred = model.predict(x)

plt.scatter(x, y)
plt.plot(x, y_pred, color="red")

plt.xlabel("Temperature")
plt.ylabel("Sales")
plt.title("Temperature vs Sales")

plt.show()
