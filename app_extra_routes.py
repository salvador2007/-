from flask import render_template
from app import app, DB_PATH
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

@app.route('/correlation')
def correlation():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM analysis_results", conn)
        conn.close()

        # 필요한 열만 남기고, 결측치 제거
        cols = ["age", "emotion", "face_shape", "genres"]
        for col in cols:
            if col not in df.columns:
                df[col] = None
        df_clean = df[cols].dropna()

        if df_clean.empty:
            return render_template("correlation.html", correlation_plot="")

        # 범주형 변수 원-핫 인코딩
        df_encoded = pd.get_dummies(df_clean, columns=["emotion", "face_shape", "genres"])

        # 상관계수 계산
        corr = df_encoded.corr().round(2)

        # 한글 폰트 설정
        import matplotlib
        import matplotlib.font_manager as fm
        available_fonts = {f.name for f in fm.fontManager.ttflist}
        if 'Malgun Gothic' in available_fonts:
            matplotlib.rcParams['font.family'] = 'Malgun Gothic'
        else:
            matplotlib.rcParams['font.family'] = 'DejaVu Sans'
        matplotlib.rcParams['axes.unicode_minus'] = False

        # 히트맵 그리기
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", cbar=True)
        plt.title("얼굴형/감정/장르 상관관계 히트맵", fontsize=14)
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()

        return render_template("correlation.html", correlation_plot=plot_data)
    except Exception as e:
        print(f"Correlation error: {e}")
        return render_template("correlation.html", correlation_plot="")


@app.route('/cluster')
def cluster_view():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT embedding FROM analysis_results", conn)
        conn.close()
        df = df.dropna()
        if df.empty:
            return render_template("cluster.html", cluster_plot="")

        embeddings = df['embedding'].apply(lambda x: [float(i) for i in x.split(',') if i])
        X = np.array(embeddings.tolist())
        n_clusters = 3 if len(X) >= 3 else max(1, len(X))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(8,6))
        scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis')
        plt.title('임베딩 클러스터링(PCA 2D)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()
        return render_template("cluster.html", cluster_plot=plot_data)
    except Exception as e:
        print(f"Cluster error: {e}")
        return render_template("cluster.html", cluster_plot="")


@app.route('/train_mlp')
def train_mlp():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT embedding, genres FROM analysis_results", conn)
        conn.close()
        df = df.dropna()
        if df.empty:
            return render_template("mlp.html", accuracy=None)

        X = []
        y = []
        for _, row in df.iterrows():
            emb = [float(i) for i in row['embedding'].split(',') if i]
            if not emb or len(emb) < 5:
                continue
            genre = str(row['genres']).split(',')[0].strip()
            if not genre:
                continue
            X.append(emb)
            y.append(genre)

        if len(X) < 5:
            return render_template("mlp.html", accuracy=None)

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=42)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        acc = round(acc * 100, 2)
        return render_template("mlp.html", accuracy=acc)
    except Exception as e:
        print(f"MLP train error: {e}")
        return render_template("mlp.html", accuracy=None)

