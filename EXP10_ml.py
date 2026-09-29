import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
data = pd.read_csv(r"C:/Users/24ucs173/Downloads/diabetes_dataset_100(1).csv")

print("DATASET:")
print(data.head(40).to_string(index=False))

X = data[["Age", "BMI", "BloodPressure"]]
y = data["Diabetes"]
model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=5,
    random_state=42
)

model.fit(X, y)
age = float(input("Enter Age: "))
bmi = float(input("Enter BMI: "))
bp = float(input("Enter Blood Pressure: "))
user_data = pd.DataFrame(
    [[age, bmi, bp]],
    columns=["Age", "BMI", "BloodPressure"]
)

prediction = model.predict(user_data)

if prediction[0] == 1:
    print("\nResult: Diabetes Risk")
else:
    print("\nResult: No Diabetes Risk")

accuracy = accuracy_score(y, model.predict(X))
print("Accuracy:", round(accuracy * 100, 2), "%")
