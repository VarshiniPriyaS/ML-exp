import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
data = {
"Age": [19, 21, 24, 27, 31, 23],
"Browsing History": [1, 2, 4, 7, 8, 3],
"Time Spent": [3, 6, 12, 18, 22, 9],
"Click": [0, 0, 1, 1, 1, 0]
}
df = pd.DataFrame(data)
print("----- DATASET -----")
print(df)
X = df[["Age", "Browsing History", "Time Spent"]]
y = df["Click"]
model = LogisticRegression()
model.fit(X, y)
age = float(input("
Enter Age: "))
history = float(input("Enter Browsing History: "))
time = float(input("Enter Time Spent on Website: "))
new_user = [[age, history, time]]
prediction = model.predict(new_user)
print("
Predicted Click:", prediction[0])
if prediction[0] == 1:
print("User is likely to CLICK the advertisement")
else:
print("User is NOT likely to CLICK the advertisement")
print("
----- FEATURE IMPORTANCE -----")
features = ["Age", "Browsing History", "Time Spent"]
for feature, coefficient in zip(features, model.coef_[0]):
print(feature, ":", coefficient)
plt.bar(features, model.coef_[0])
plt.xlabel("Features")
plt.ylabel("Coefficient")
plt.title("Feature Importance - Logistic Regression")
plt.xticks(rotation=20)
plt.grid(True)
plt.show()
