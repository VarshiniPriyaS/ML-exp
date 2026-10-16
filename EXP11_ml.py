import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
df = pd.read_csv(r"C:/Users/24ucs173/Downloads/customer.csv")
x = df[["Age", "items", "spending"]]
k = KMeans(n_clusters=3, random_state=0, n_init=10)
df["cluster"] = k.fit_predict(x)
print(df)
print("\nCluster centers:")
print(k.cluster_centers_)
plt.scatter(df["Age"], df["spending"], c=df["cluster"], s=100, cmap="viridis")
plt.xlabel("Age")
plt.ylabel("Spending")
plt.title("Customer Segmentation Using K-Means Clustering")
plt.colorbar(label="Cluster")
plt.show()
