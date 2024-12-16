from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

app = Flask(__name__)

#load data
df = pd.read_csv("investasi_saham.csv")

#pemilihan vitur
features = ['open_price', 'close', 'volume', 'frequency', 'foreign_buy', 'foreign_sell']
df = df[features]

#handling missing values
df.fillna(df.mean(), inplace=True)

#standarisasi data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

#membagi data menajdi data latih dan data uji (80% latih dan 20% data uji)
X_train, X_test = train_test_split(df_scaled, test_size=0.2, random_state=42)

#menentukan jumlah clustering optimal (menggunakan metode elbow)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)

#menampilkan hasil berupa grafik
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#memilih jumlah clustering berdasarkan plot elbow
n_clusters = 3

#membuat model k-means dan latih dengan data latih
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
kmeans.fit(X_train)

#memprediksi label clustering untuk data uji
y_pred = kmeans.predict(X_test)

#evaluasi model menggunakan silhouette coefficient
silhouette_avg = silhouette_score(X_test, y_pred)
print("Shilhouette coefficient:", silhouette_avg)

#Visualisasi hasil clustering (pada data uji)
plt.scatter (X_test[:, 0], X_test[:, 1], c=y_pred, s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')
plt.title('Clusters of Saham')
plt.show()

@app.route('/proses_data', methods=['POST'])
def proses_data():
    #ambil data inputan dari form
    harga_awal = float(request.form['harga_awal'])
    harga_akhir = float(request.form['harga_akhir'])
    volume_pasar = float(request.form['volume_pasar'])

    #ubah data inputan menjadi format array
    data_baru = np.array([[harga_awal, harga_akhir, volume_pasar]])

    #standarisasi data inputan
    data_baru_scaled = scaler.transform(data_baru)

    #prediksi cluster untuk data input baru
    y_pred = kmeans.predict(data_baru_scaled)

    #clustering yang diprediksi
    cluster_terprediksi = y_pred[0]

    #menampilkan hasil prediksi ke template html
    return render_template('hasil_uji.html', cluster=cluster_terprediksi)