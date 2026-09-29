import numpy as np
import matplotlib.pyplot as plt

x = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [4, 3],
    [3, 4]
])

y = np.array([0, 0, 0, 1, 1])

w = np.array([0.0, 0.0])
b = 0.0
lr = 1.0

print("--- Training Dataset ---")

for i in range(len(x)):
    label = "Spam" if y[i] == 1 else "Not Spam"
    print(
        f"Email {i+1}: Features "
        f"(Keywords: {x[i][0]}, Lines: {x[i][1]}) "
        f"-> Label: {label}"
    )

print("-" * 24)

for epoch in range(10):
    errors = 0

    for i in range(len(x)):
        z = np.dot(x[i], w) + b

        pred = 1 if z >= 0 else 0

        update = lr * (y[i] - pred)

        w = w + update * x[i]
        b = b + update

        if update != 0:
            errors += 1

    if errors == 0:
        break

print("\nModel training complete.")
print("Final Weights:", w)
print("Final Bias:", b)

print("\n--- Test a New Email ---")

try:
    keywords = float(input("Enter the number of spam keywords: "))
    lines = float(input("Enter the number of links: "))

    new_input = np.array([keywords, lines])

    z_new = np.dot(new_input, w) + b

    pred_new = 1 if z_new >= 0 else 0

    result = "Spam" if pred_new == 1 else "Not Spam"

    print(
        f"\nPrediction for input "
        f"[{keywords}, {lines}]: {result}"
    )

    plt.figure(figsize=(8, 6))

    plt.scatter(
        x[y == 0, 0],
        x[y == 0, 1],
        marker='o',
        s=100,
        label='Not Spam'
    )

    plt.scatter(
        x[y == 1, 0],
        x[y == 1, 1],
        marker='x',
        s=100,
        label='Spam'
    )

    plt.scatter(
        keywords,
        lines,
        marker='*',
        color='gold',
        s=200,
        edgecolor='black',
        label='User Input'
    )

    x_values = np.linspace(0, 6, 100)

    if w[1] != 0:
        y_values = -(w[0] * x_values + b) / w[1]

        plt.plot(
            x_values,
            y_values,
            label='Decision Boundary',
            color='red'
        )

    else:
        x_boundary = -b / w[0]

        plt.axvline(
            x_boundary,
            label='Decision Boundary',
            color='red'
        )

    plt.xlabel("Number of spam keywords")
    plt.ylabel("Number of links")
    plt.title("Prediction classification")
    plt.legend()
    plt.grid(True)
    plt.show()

except ValueError:
    print("Invalid input. Please enter numbers only.")
