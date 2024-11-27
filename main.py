import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import scipy.cluster.hierarchy as shc
import seaborn as sns

sns.set_style("darkgrid")

# Load the dataset (replace the path if necessary)
try:
    data = pd.read_csv(r"C:\Users\Shreya Nithin\Desktop\ML\wine_data.csv")

    if data.empty:
        print("Error: The dataset is empty.")
        exit()
except FileNotFoundError:
    print("Error: The specified file path for 'wine_data.csv' is not found. Please check and try again.")
    exit()

# Selecting relevant features
X = data.iloc[:, :].values

# Splitting the dataset into training and test data (80% train, 20% test)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)

# Standardizing the features for the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Use PCA for dimensionality reduction
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Agglomerative Clustering: Dendrogram
plt.figure(figsize=(10, 7))
plt.title('Agglomerative Clustering Dendrogram', fontsize=16)
dendrogram = shc.dendrogram(shc.linkage(X_train_pca, method='ward'))
plt.show()

# Apply Agglomerative Clustering based on the dendrogram
elbow_point = 3  # Set based on dendrogram (can be adjusted)
agg_clust = AgglomerativeClustering(n_clusters=elbow_point, linkage='ward')
y_agg = agg_clust.fit_predict(X_train_pca)

# Using the elbow method to find the optimal number of clusters for KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_train_pca)
    wcss.append(kmeans.inertia_)

# Calculate the optimal number of clusters for KMeans
elbow_point_kmeans = np.argmax(np.diff(np.diff(wcss))) + 2  # Adjusted elbow point

# Plot the WCSS and add the elbow point with a red circle boundary
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', markersize=5, color='b', label="WCSS (Within-Cluster Sum of Squares)")
plt.axvline(x=elbow_point_kmeans, color='r', linestyle='--', label=f'Elbow Point (K={elbow_point_kmeans})')

# Highlight the elbow point with a red circle boundary
plt.scatter(elbow_point_kmeans, wcss[elbow_point_kmeans-1], color='r', s=100, edgecolor='black', zorder=5, label="Elbow Point (highlighted)")

plt.title('The Elbow Method for Optimal KMeans Clusters', fontsize=14)
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.legend()
plt.show()

# Applying KMeans with optimal clusters from elbow method
kmeans = KMeans(n_clusters=elbow_point_kmeans, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(X_train_pca)
y_kmeans = kmeans.predict(X_train_pca)

# Visualizing KMeans clusters with reduced point size for clarity
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[y_kmeans == 0, 0], X_train_pca[y_kmeans == 0, 1], s=50, c='red', label='Cluster 1')
for cluster_num in range(1, elbow_point_kmeans):
    plt.scatter(X_train_pca[y_kmeans == cluster_num, 0], X_train_pca[y_kmeans == cluster_num, 1], s=50, label=f'Cluster {cluster_num + 1}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow', edgecolors='k', label='Centroids')
plt.title(f'Clusters of Wine Data (KMeans with {elbow_point_kmeans} clusters)', fontsize=14)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Calculate the maximum distance from the cluster centers in the training data
max_distance = max([np.linalg.norm(point - kmeans.cluster_centers_[label]) for point, label in zip(X_train_pca, y_kmeans)])
print(f"Maximum distance from cluster centers (training data): {max_distance:.4f}")

# User prompt for specific test data row for accuracy as distance to cluster center
print(f"Please enter the serial number (index) of the test data for prediction (valid range: 0 to {len(X_test) - 1}):")
try:
    test_data_index = int(input("Enter the serial number of the test data row (integer value): "))
    if 0 <= test_data_index < len(X_test):
        # Select and process the specific test data
        specific_test_data = X_test[test_data_index]
        specific_test_data_scaled = scaler.transform([specific_test_data])
        specific_test_data_pca = pca.transform(specific_test_data_scaled)
        
        # Predict the cluster for this data point
        specific_test_prediction = kmeans.predict(specific_test_data_pca)
        cluster_center = kmeans.cluster_centers_[specific_test_prediction[0]]
        distance_to_center = np.linalg.norm(specific_test_data_pca - cluster_center)
        
        # Display the distance and accuracy as a percentage
        accuracy_percentage = (1 - (distance_to_center / max_distance)) * 100
        print(f"Predicted cluster for selected test data (index {test_data_index}): {specific_test_prediction[0] + 1}")
        print(f"Distance to cluster center: {distance_to_center:.4f}")
        print(f"Accuracy for this specific data point: {accuracy_percentage:.2f}%")
    else:
        print("Error: Invalid serial number, please choose a valid row index.")
except ValueError:
    print("Error: Please enter a valid integer.")

# New feature: Ask user to input a cluster number to test accuracy (1-10)
try:
    cluster_input = int(input("Please enter a cluster number (1-10) to check accuracy: "))
    if 1 <= cluster_input <= 10:
        # Applying KMeans with the user-specified number of clusters
        kmeans_user = KMeans(n_clusters=cluster_input, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans_user.fit(X_train_pca)
        y_kmeans_user = kmeans_user.predict(X_train_pca)

        # Visualize the clusters again with the new number of clusters
        plt.figure(figsize=(8, 6))
        for cluster_num in range(cluster_input):
            plt.scatter(X_train_pca[y_kmeans_user == cluster_num, 0], X_train_pca[y_kmeans_user == cluster_num, 1], s=50, label=f'Cluster {cluster_num + 1}')
        plt.scatter(kmeans_user.cluster_centers_[:, 0], kmeans_user.cluster_centers_[:, 1], s=200, c='yellow', edgecolors='k', label='Centroids')
        plt.title(f'Clusters of Wine Data (KMeans with {cluster_input} clusters)', fontsize=14)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.show()

        # Recalculate maximum distance with the new number of clusters
        max_distance_user = max([np.linalg.norm(point - kmeans_user.cluster_centers_[label]) for point, label in zip(X_train_pca, y_kmeans_user)])
        print(f"Maximum distance from cluster centers (user-defined clusters): {max_distance_user:.4f}")

        # Predict for a test data point and calculate accuracy
        if 0 <= test_data_index < len(X_test):
            specific_test_data_user = X_test[test_data_index]
            specific_test_data_scaled_user = scaler.transform([specific_test_data_user])
            specific_test_data_pca_user = pca.transform(specific_test_data_scaled_user)
            
            specific_test_prediction_user = kmeans_user.predict(specific_test_data_pca_user)
            cluster_center_user = kmeans_user.cluster_centers_[specific_test_prediction_user[0]]
            distance_to_center_user = np.linalg.norm(specific_test_data_pca_user - cluster_center_user)
            accuracy_percentage_user = (1 - (distance_to_center_user / max_distance_user)) * 100

            print(f"Predicted cluster for selected test data (index {test_data_index}) with {cluster_input} clusters: {specific_test_prediction_user[0] + 1}")
            print(f"Distance to cluster center: {distance_to_center_user:.4f}")
            print(f"Accuracy for this specific data point with {cluster_input} clusters: {accuracy_percentage_user:.2f}%")
    else:
        print("Error: Please enter a valid cluster number between 1 and 10.")
except ValueError:
    print("Error: Please enter a valid integer.")