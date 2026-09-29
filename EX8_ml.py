import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
X = np.array([
    [20, 80],
    [25, 85],
    [30, 90],
    [50, 130],
    [55, 140]
])
y = np.array([0, 0, 0, 1, 1])
model = SVC(kernel='linear', C=1.0)
model.fit(X, y)
y_pred = model.predict(X)
print("Actual Output: ", y)
print("Predicted Output:", y_pred)
accuracy = accuracy_score(y, y_pred) * 100
print("\nAccuracy:", accuracy, "%")
print("\nSupport Vectors:")
print(model.support_vectors_)
print("\nWeights:", model.coef_[0])
print("Bias:", model.intercept_[0])
age = float(input("\nEnter Age: "))
bp = float(input("Enter Blood Pressure Level: "))
new_patient = np.array([[age, bp]])
prediction = model.predict(new_patient)
if prediction[0] == 0:
    print("Predicted Class: No Condition")
else:
    print("Predicted Class: Condition")
plt.figure(figsize=(8, 6))
plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    marker='o',
    s=100,
    label='No Condition'
)
plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    marker='x',
    s=100,
    label='Condition'
)
plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=200,
    facecolors='none',
    edgecolors='black',
    label='Support Vectors'
)
x_values = np.linspace(15, 60, 100)
w = model.coef_[0]
b = model.intercept_[0]
y_values = -(w[0] * x_values + b) / w[1]
plt.plot(
    x_values,
    y_values,
    label='Decision Boundary'
)
plt.scatter(
    age,
    bp,
    marker='*',
    s=200,
    label='New Patient'
)
plt.xlabel("Age")
plt.ylabel("Blood Pressure Level")
plt.title("SVM Classification")
plt.legend()
plt.grid(True)
plt.show()
