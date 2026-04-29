import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. DATA GENERATION (Simulating 50 features of 500 Bank Customers)
np.random.seed(42)
data = np.random.rand(500, 50)
feature_names = [f'Metric_{i}' for i in range(50)]
df = pd.DataFrame(data, columns=feature_names)

# 2. PRE-PROCESSING (Crucial for Math Stability)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. PCA (Dimensionality Reduction)
# We keep 3 components so we can plot in 3D
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# 4. K-MEANS WITH SILHOUETTE SCORE (Automated K Selection)
# We test K from 2 to 6 and pick the one with the highest "Clarity" (Silhouette)
scores = []
k_range = range(2, 7)
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_pca)
    scores.append(silhouette_score(X_pca, labels))

best_k = k_range[np.argmax(scores)]
print(f"Optimal Clusters detected via Silhouette Math: {best_k}")

# Final Model
kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# 5. OUTPUT 1: THE 3D CLUSTER PLOT
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=clusters, cmap='viridis', s=40)
ax.set_title(f"Archetype-X: {best_k} Segments Found")
plt.colorbar(scatter)
plt.show()

# 6. OUTPUT 2: FEATURE IMPORTANCE (The "Why")
# We look at the 'loadings' (weights) of the first Principal Component
loadings = pd.DataFrame(
    pca.components_[0],
    index=feature_names,
    columns=['Weight']
).abs().sort_values(by='Weight', ascending=False)

plt.figure(figsize=(10, 5))
loadings.head(10).plot(kind='bar', color='teal')
plt.title("Top 10 Drivers of User Behavior")
plt.ylabel("Mathematical Influence (Loading)")
plt.xticks(rotation=45, ha='right') # Rotates labels by 45 degrees
plt.tight_layout()                 # Prevents labels from getting cut off
plt.show()
