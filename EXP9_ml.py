import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
df = pd.read_csv(r"C:/Users/24ucs173/Downloads/student_attendance_100.csv")
print("DATASET:")
print(df)
print(df.head(40).to_string(index=False))
X = df[["Attendance", "StudyHours",
        "InternalMarks", "Participation"]]
y = df["Result"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nActual Result:")
print(list(y_test))
print("\nPredicted Result:")
print(list(y_pred))
print("\nAccuracy:", accuracy * 100, "%")
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Pass", "Fail"]
)
print("\nConfusion Matrix:")
print(cm)
print("\n--- STUDENT PREDICTION ---")
attendance = float(input("Enter Attendance (%): "))
study_hours = float(input("Enter Study Hours: "))
internal_marks = float(input("Enter Internal Marks: "))
participation = float(input("Enter Participation (1-10): "))
student = pd.DataFrame({
    "Attendance": [attendance],
    "StudyHours": [study_hours],
    "InternalMarks": [internal_marks],
    "Participation": [participation]
})
prediction = model.predict(student)
print("\nStudent Result:", prediction[0])
plt.figure(figsize=(8, 5))
plt.bar(
    ["Attendance", "Study Hours",
     "Internal Marks", "Participation"],
    [attendance, study_hours, internal_marks, participation]
)
plt.xlabel("Student Features")
plt.ylabel("Values")
plt.title("Student Input")
plt.tight_layout()
plt.show()




