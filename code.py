import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv'
data = pd.read_csv(file_path)

# Select relevant features
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Number of clusters
num_clusters = 5

# Initialize K-means model with explicit n_init
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)

# Fit the model to the scaled data
kmeans.fit(X_scaled)

# Predict the clusters
cluster_labels = kmeans.labels_

# Add cluster labels to the original dataset
data['Cluster'] = cluster_labels

# Visualize the clusters
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=data, palette='viridis')
plt.title('K-means Clustering of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
