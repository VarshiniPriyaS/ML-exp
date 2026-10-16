import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
df = pd.read_csv(r'C:/Users/24ucs173/Downloads/products_hierarchical.csv')
X = df[['Sales', 'Reviews', 'Price', 'Features']]
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
df['cluster'] = model.fit_predict(X)
print(df)
Z = linkage(X, method='ward')
plt.figure(figsize=(10, 6))
dendrogram(Z, labels=df['product'].values)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Products")
plt.ylabel("Distance")
plt.show()


