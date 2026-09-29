import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

data={
    'Age':[22,27,31,36,42,48,53,59],
    'Gender':[1,0,1,0,1,0,1,0],
    'BMI':[21.8,24.2,26.7,28.5,30.1,32.4,33.8,35.2],
    'Bloodpressure':[115,122,128,134,141,147,152,158],
    'Cholesterol':[170,185,195,210,225,240,255,270],
    'Disease':[0,0,0,1,1,1,1,1]
    }
df=pd.DataFrame(data)
feature_names = ['Age','Gender','BMI','Bloodpressure','Cholesterol']
X=df[feature_names]
y=df['Disease']
model=LogisticRegression()
model.fit(X,y)

age=int(input("Enter Age:"))
gender=int(input("Enter Gender(0=female,1=Male):"))
bmi=float(input("Enter BMI:"))
bp=int(input("Enter BloodPressure:"))
chol=int(input("Enter Cholesterol:"))

test=pd.DataFrame([[age,gender,bmi,bp,chol]], columns=feature_names)
result=model.predict(test)

if result[0]==1:
    print("Disease Detected")
else:
    print("No Disease")

X_line=np.linspace(df['Age'].min(),df['Age'].max(),100)
gender_mean=df['Gender'].mean()
bmi_mean=df['BMI'].mean()
bp_mean=df['Bloodpressure'].mean()
chol_mean=df['Cholesterol'].mean()

X_plot=np.column_stack((X_line,np.full_like(X_line,gender_mean),np.full_like(X_line,bmi_mean),np.full_like(X_line,bp_mean),np.full_like(X_line,chol_mean)))
X_plot_df=pd.DataFrame(X_plot, columns=feature_names)
y_prob=model.predict_proba(X_plot_df)[:,1]

plt.scatter(df['Age'],df['Disease'],color='blue',label='data')
plt.plot(X_line,y_prob,color='red',linewidth=2,label='Logistic Regression Curve')
plt.xlabel("Age")
plt.ylabel("Probability of Disease")
plt.title("Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()

