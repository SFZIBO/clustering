import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
data_path = "dataset/country_wise_latest.csv"
df = pd.read_csv(data_path)

# Pilih fitur yang akan digunakan
features = ["Confirmed", "Deaths", "Recovered", "Active"]
X = df[features]

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Menentukan jumlah cluster dengan metode Elbow
inertia = []
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot hasil Elbow Method
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('Inertia')
plt.title('Metode Elbow untuk Menentukan K Optimal')
plt.show()

# Menggunakan K optimal (misalnya, K=3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Menampilkan hasil cluster
print(df[["Country/Region", "Confirmed", "Deaths", "Recovered", "Active", "Cluster"]])

# Elbow plot
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('Inertia')
plt.title('Metode Elbow untuk Menentukan K Optimal')
plt.savefig("elbow_plot.png")
plt.show()

# Scatter hasil clustering
sns.scatterplot(x=df['Confirmed'], y=df['Active'], hue=df['Cluster'], palette='viridis')
plt.title('Hasil Clustering Negara Berdasarkan Data COVID-19')
plt.xlabel('Confirmed Cases')
plt.ylabel('Active Cases')
plt.savefig("cluster_plot.png") 
plt.show()
