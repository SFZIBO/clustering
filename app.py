from flask import Flask, render_template, request, send_from_directory
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ELBOW_PLOTS'] = 'static/results/elbow_plots'
app.config['CLUSTER_PLOTS'] = 'static/results/cluster_plots'

# Buat folder jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ELBOW_PLOTS'], exist_ok=True)
os.makedirs(app.config['CLUSTER_PLOTS'], exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected")
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Proses clustering
            df = pd.read_csv(filepath)
            features = ["Confirmed", "Deaths", "Recovered", "Active"]
            X = df[features]
            
            # Normalisasi
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Elbow method
            inertia = []
            k_range = range(1, 10)
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_scaled)
                inertia.append(kmeans.inertia_)
            
            # Tentukan K optimal (contoh sederhana)
            optimal_k = 3
            
            # Plot elbow
            plt.figure(figsize=(8, 5))
            plt.plot(k_range, inertia, marker='o')
            plt.xlabel('Number of Clusters (K)')
            plt.ylabel('Inertia')
            plt.title('Elbow Method For Optimal K')
            elbow_plot = f"elbow_{filename.split('.')[0]}.png"
            elbow_path = os.path.join(app.config['ELBOW_PLOTS'], elbow_plot)
            plt.savefig(elbow_path)
            plt.close()
            
            # Clustering dengan K optimal
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            df['Cluster'] = kmeans.fit_predict(X_scaled)
            
            # Plot hasil clustering
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df['Confirmed'], y=df['Active'], hue=df['Cluster'], palette='viridis')
            plt.title('Country Clustering Based on COVID-19 Data')
            plt.xlabel('Confirmed Cases')
            plt.ylabel('Active Cases')
            cluster_plot = f"cluster_{filename.split('.')[0]}.png"
            cluster_path = os.path.join(app.config['CLUSTER_PLOTS'], cluster_plot)
            plt.savefig(cluster_path)
            plt.close()
            
            # Simpan hasil clustering
            cluster_data = f"cluster_results_{filename}"
            cluster_data_path = os.path.join(app.config['UPLOAD_FOLDER'], cluster_data)
            df.to_csv(cluster_data_path, index=False)
            
            # Ambil sample data untuk ditampilkan
            sample_data = df.sample(min(10, len(df))).to_dict('records')
            
            return render_template('index.html', results={
                'elbow_plot': elbow_plot,
                'cluster_plot': cluster_plot,
                'optimal_k': optimal_k,
                'sample_data': sample_data,
                'cluster_data': cluster_data
            })
            
        except Exception as e:
            return render_template('index.html', error=f"Error processing file: {str(e)}")
    
    return render_template('index.html', error="Invalid file format. Please upload a CSV file.")

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)