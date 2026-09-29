import math
import matplotlib.pyplot as plt

data = [
    [30, 115, 170, "no"],
    [42, 125, 190, "no"],
    [55, 145, 250, "yes"],
    [38, 120, 180, "no"],
    [62, 155, 270, "yes"],
    [48, 135, 220, "yes"],
    [35, 118, 175, "no"],
    [58, 150, 260, "yes"]
]
new = [52, 142, 245]
k = 3
dist = []

for x in data:
    d = math.sqrt((x[0] - new[0])**2 + (x[1] - new[1])**2 + (x[2] - new[2])**2)
    dist.append((d, x))

dist.sort()
nearest = dist[:k]

print("k nearest neighbours:")
for d, x in nearest:
    print(f"distance={d:.2f}->{x[3]}")

yes = sum(x[1][3] == "yes" for x in nearest)
no = sum(x[1][3] == "no" for x in nearest)

print("\nyes=", yes)
print("no=", no)

prediction = "yes" if yes > no else "no"
print("prediction=", prediction)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

for x in data:
    label_val = x[3]
    ax.scatter(x[0], x[1], x[2], marker="x", s=80, label=label_val)

for d, x in nearest:
    ax.plot(
        [new[0], x[0]], [new[1], x[1]], [new[2], x[2]], "--", linewidth=1.5
    )

ax.scatter(
    new[0], new[1], new[2], marker="*", s=300, label="new patient"
)

ax.set_xlabel("age")
ax.set_ylabel("blood pressure")
ax.set_zlabel("cholesterol")
ax.set_title("k-nn classification (k=3)")

handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys())

plt.show()
