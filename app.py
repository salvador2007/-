from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory, send_file
import sqlite3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import hashlib
from datetime import datetime
import tempfile
import logging
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
# --- 얼굴형 분류 모델 준비 (app.py 상단에) ---
import json
from tensorflow import keras
import numpy as np
from PIL import Image

# 모델 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')

FACE_MODEL_JSON = os.path.join(MODEL_DIR, 'face_shape_optimized_model_architecture.json')
FACE_MODEL_WEIGHTS = os.path.join(CHECKPOINT_DIR, 'best_face_shape_optimized_model_01_0.1990.weights.h5')
CLASS_INDICES = os.path.join(MODEL_DIR, 'class_indices.json')

# 기본 얼굴형 라벨 (class_indices.json이 없을 때 사용)
DEFAULT_FACE_SHAPE_CLASSES = [
    "Heart",
    "Long",
    "Oblong",
    "Oval",
    "Round",
    "Square",
    "Triangle",
]

# 얼굴형 분류 모델 버전 정보 (논문/보고서/시스템 설명용)
FACE_SHAPE_MODEL_VERSION = "5.0"

# 전역 모델, 클래스명
face_shape_model = None
face_shape_classes = []

def load_face_shape_model():
    """얼굴형 분류 모델 로드 (.keras 우선, 없으면 json+weights fallback)"""
    global face_shape_model, face_shape_classes
    import json
    try:
        keras_path = os.path.join(MODEL_DIR, 'face_shape_optimized_model.keras')
        if os.path.exists(keras_path):
            try:
                face_shape_model = keras.models.load_model(keras_path, compile=False)
                logger.info(f"얼굴형 분류 모델(.keras, compile=False) 로드 성공: {keras_path}")
            except Exception as e:
                logger.error(f"얼굴형 분류 모델(.keras, compile=False) 로드 실패: {e}")
                face_shape_model = None
        else:
            # fallback: json+weights 여러 조합 시도
            weights_candidates = [
                os.path.join(CHECKPOINT_DIR, 'best_face_shape_optimized_model_01_0.1990.weights.h5'),
                os.path.join(CHECKPOINT_DIR, 'best_face_shape_optimized_model_02_0.2010.weights.h5'),
                os.path.join(MODEL_DIR, 'face_shape_optimized_model.weights.h5'),
                os.path.join(MODEL_DIR, 'face_shape_optimized_model_emergency.weights.h5'),
            ]
            loaded = False
            if os.path.exists(FACE_MODEL_JSON):
                with open(FACE_MODEL_JSON, 'r', encoding='utf-8') as f:
                    model_json = f.read()
                for weights_path in weights_candidates:
                    if os.path.exists(weights_path):
                        try:
                            model = keras.models.model_from_json(model_json)
                            model.load_weights(weights_path)
                            face_shape_model = model
                            logger.info(f"얼굴형 분류 모델(json+weights) 로드 성공: {FACE_MODEL_JSON}, {weights_path}")
                            loaded = True
                            break
                        except Exception as e2:
                            logger.error(f"얼굴형 분류 모델(json+weights) 로드 실패: {weights_path}, {e2}")
                if not loaded:
                    logger.error("얼굴형 분류 모델 json+weights 조합이 모두 실패했습니다.")
                    face_shape_model = None
            else:
                logger.error("얼굴형 분류 모델 구조(json) 파일이 존재하지 않습니다.")
                face_shape_model = None
    except Exception as e:
        logger.error(f"얼굴형 분류 모델(.keras) 로드 실패: {e}")
        face_shape_model = None
    # 클래스 인덱스 로드
    try:
        if os.path.exists(CLASS_INDICES):
            with open(CLASS_INDICES, 'r', encoding='utf-8') as f:
                idx_map = json.load(f)
            face_shape_classes.clear()
            for k, v in sorted(idx_map.items(), key=lambda x: x[1]):
                face_shape_classes.append(k)
        else:
            face_shape_classes.clear()
            face_shape_classes.extend(DEFAULT_FACE_SHAPE_CLASSES)
    except Exception as e:
        logger.error(f"Face shape class index load error: {e}")
        face_shape_classes.clear()
        face_shape_classes.extend(DEFAULT_FACE_SHAPE_CLASSES)
    # 최종 상태 로그
    if face_shape_model is not None:
        logger.info("Face shape model loaded and ready.")
    else:
        logger.error("Face shape model is NOT ready. 얼굴형 분류 기능이 비활성화됩니다.")

def predict_face_shape(img_path):
    if face_shape_model is None:
        return "Unknown", 0.0
    try:
        # 이미지 불러오기 & 전처리 (224x224, RGB, 정규화)
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        preds = face_shape_model.predict(arr)
        idx = int(np.argmax(preds))
        prob = float(np.max(preds))
        label = face_shape_classes[idx] if idx < len(face_shape_classes) else "Unknown"
        return label, prob
    except Exception as e:
        logger.error(f"얼굴형 예측 오류: {e}")
        return "Unknown", 0.0

# DeepFace 임포트 및 에러 메시지 저장
deepface_import_error = None
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except Exception as e:
    DEEPFACE_AVAILABLE = False
    deepface_import_error = str(e)
    logging.warning(f"DeepFace not available. Face analysis will use mock data. ImportError: {e}")

app = Flask(__name__)

# 보안 강화된 설정
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 최대 파일 크기
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# 설정
DB_PATH = os.environ.get('DB_PATH', 'analysis_data.db')
ADMIN_PASSWORD_HASH = generate_password_hash(os.environ.get('ADMIN_PASSWORD', 'admin1234'))
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 업로드 폴더 생성
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, mode=0o755)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 모델 로드는 로깅 설정 이후에 수행한다
load_face_shape_model()

# 한글 폰트 설정 (에러 방지)
import matplotlib.font_manager as fm
available_fonts = {f.name for f in fm.fontManager.ttflist}
if 'Malgun Gothic' in available_fonts:
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def init_db():
    """데이터베이스 초기화 및 보안 설정"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # 테이블 생성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                age INTEGER CHECK(age >= 0 AND age <= 150),
                gender TEXT CHECK(gender IN ('Man', 'Woman', 'Unknown')),
                gender_confidence REAL CHECK(gender_confidence >= 0 AND gender_confidence <= 100),
                emotion TEXT,
                emotion_confidence REAL CHECK(emotion_confidence >= 0 AND emotion_confidence <= 100),
                emotion_scores TEXT,
                genres TEXT,
                filename_hash TEXT,
                face_shape TEXT DEFAULT 'Unknown',
                embedding TEXT,
                ip_address TEXT,
                user_agent TEXT
            )
        ''')
        # 인덱스 생성 (성능 향상)
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON analysis_results(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_emotion ON analysis_results(emotion)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_filename_hash ON analysis_results(filename_hash)')
        # 기존 DB에 embedding 컬럼이 없을 수 있으므로 추가 시도
        try:
            cursor.execute("ALTER TABLE analysis_results ADD COLUMN embedding TEXT")
        except sqlite3.OperationalError:
            pass
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        raise

def allowed_file(filename):
    """허용된 파일 확장자 확인"""
    if not filename or '.' not in filename:
        return False
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS

def validate_mime_type(file_content):
    """파일 내용 기반 MIME 타입 검증"""
    try:
        if file_content.startswith(b'\xff\xd8\xff'):  # JPEG
            return True
        elif file_content.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
            return True
        return False
    except Exception:
        return False

def get_file_hash(file_content):
    """파일 해시 생성 (보안 강화)"""
    try:
        return hashlib.sha256(file_content).hexdigest()[:16]
    except Exception as e:
        logger.error(f"Hash generation error: {e}")
        return "unknown"

def get_client_ip():
    """클라이언트 IP 주소 안전하게 가져오기"""
    if request.environ.get('HTTP_X_FORWARDED_FOR'):
        ip = request.environ['HTTP_X_FORWARDED_FOR'].split(',')[0].strip()
    else:
        ip = request.environ.get('REMOTE_ADDR', 'unknown')
    return ip[:45]  # IPv6 최대 길이 제한

def sanitize_user_agent():
    """User Agent 문자열 정리"""
    user_agent = request.headers.get('User-Agent', 'unknown')
    return user_agent[:500]  # 길이 제한

def analyze_face(image_path):
    """얼굴 분석 함수 (DeepFace 사용)"""
    if not DEEPFACE_AVAILABLE:
        raise RuntimeError(
            "DeepFace library is not installed. Install requirements for accurate analysis."
        )
    try:
        analysis = DeepFace.analyze(
            img_path=image_path,
            actions=['age', 'gender', 'emotion'],
            enforce_detection=False,
            silent=True
        )
        if isinstance(analysis, list):
            analysis = analysis[0]
        age = max(0, min(150, int(analysis.get('age', 25))))
        gender = analysis.get('gender', 'Unknown')
        emotion = analysis.get('dominant_emotion', 'neutral')
        emotion_scores = analysis.get('emotion', {'neutral': 100})
        gender_confidence = 85.0
        if isinstance(gender, dict):
            gender_confidence = max(0, min(100, max(gender.values())))
            gender = max(gender, key=gender.get)
        if gender not in ['Man', 'Woman']:
            gender = 'Unknown'
        emotion_confidence = max(0, min(100, emotion_scores.get(emotion, 0)))
        return {
            'age': age,
            'gender': gender,
            'gender_confidence': gender_confidence,
            'emotion': emotion,
            'emotion_confidence': emotion_confidence,
            'emotion_scores': emotion_scores
        }
    except Exception as e:
        logger.error(f"Face analysis error: {e}")
        return {
            'age': 25,
            'gender': 'Unknown',
            'gender_confidence': 0,
            'emotion': 'neutral',
            'emotion_confidence': 0,
            'emotion_scores': {'neutral': 100}
        }

def get_face_embedding(image_path):
    """DeepFace 임베딩 추출"""
    if not DEEPFACE_AVAILABLE:
        return ""
    try:
        rep = DeepFace.represent(img_path=image_path, enforce_detection=False)
        if isinstance(rep, list):
            rep = rep[0]
        if isinstance(rep, dict) and 'embedding' in rep:
            rep = rep['embedding']
        vector = [float(x) for x in rep]
        return ",".join(f"{v:.6f}" for v in vector)
    except Exception as e:
        logger.error(f"Embedding extract error: {e}")
        return ""

@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"404 error for {request.url}")
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"500 error: {e}")
    return render_template('500.html'), 500

@app.errorhandler(413)
def file_too_large(e):
    flash('파일 크기가 너무 큽니다. 최대 16MB까지 업로드 가능합니다.', 'error')
    return redirect(url_for('index'))

@app.route('/health')
def health_check():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('SELECT 1')
        conn.close()
        return jsonify({
            "status": "healthy", 
            "timestamp": datetime.now().isoformat(),
            "deepface_available": DEEPFACE_AVAILABLE
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/')
def index():
    # DeepFace import 실패 시, 사용자에게 실제 에러 메시지 안내
    if not DEEPFACE_AVAILABLE:
        flash(f"DeepFace library is not installed or failed to import.<br><small>{deepface_import_error}</small>", "error")
    return render_template('index.html')

def create_genre_plot(df):
    try:
        # 한글 폰트 설정 (NanumGothic 우선 적용)
        import matplotlib
        import matplotlib.font_manager as fm
        available_fonts = {f.name for f in fm.fontManager.ttflist}
        if 'NanumGothic' in available_fonts:
            matplotlib.rcParams['font.family'] = 'NanumGothic'
        elif 'Malgun Gothic' in available_fonts:
            matplotlib.rcParams['font.family'] = 'Malgun Gothic'
        elif 'AppleGothic' in available_fonts:
            matplotlib.rcParams['font.family'] = 'AppleGothic'
        else:
            matplotlib.rcParams['font.family'] = 'DejaVu Sans'
        matplotlib.rcParams['axes.unicode_minus'] = False

        genre_data = []
        for genres_str in df['genres'].dropna():
            if genres_str and len(str(genres_str)) < 1000:
                genre_data.extend([g.strip() for g in str(genres_str).split(',') if len(g.strip()) < 100])
        if not genre_data:
            return ""
        genre_counts = pd.Series(genre_data).value_counts().head(20)
        plt.figure(figsize=(10, 6))
        genre_counts.plot(kind='bar', color='skyblue')
        plt.title('장르별 선호도', fontsize=14)
        plt.xlabel('장르')
        plt.ylabel('선택 횟수')
        plt.xticks(rotation=45)
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()
        return plot_data
    except Exception as e:
        logger.error(f"Genre plot error: {e}")
        return ""

def create_emotion_plot(df):
    try:
        # 한글 폰트 설정 (NanumGothic 우선 적용)
        import matplotlib
        import matplotlib.font_manager as fm
        available_fonts = {f.name for f in fm.fontManager.ttflist}
        if 'NanumGothic' in available_fonts:
            matplotlib.rcParams['font.family'] = 'NanumGothic'
        elif 'Malgun Gothic' in available_fonts:
            matplotlib.rcParams['font.family'] = 'Malgun Gothic'
        elif 'AppleGothic' in available_fonts:
            matplotlib.rcParams['font.family'] = 'AppleGothic'
        else:
            matplotlib.rcParams['font.family'] = 'DejaVu Sans'
        matplotlib.rcParams['axes.unicode_minus'] = False

        emotion_genre_data = []
        for _, row in df.iterrows():
            if (pd.notna(row['emotion']) and pd.notna(row['genres']) and 
                len(str(row['genres'])) < 1000):
                genres = [g.strip() for g in str(row['genres']).split(',') if len(g.strip()) < 100]
                for genre in genres[:10]:
                    emotion_genre_data.append({
                        'emotion': str(row['emotion'])[:50],
                        'genre': genre
                    })
        if not emotion_genre_data:
            return ""
        emotion_df = pd.DataFrame(emotion_genre_data)
        crosstab = pd.crosstab(emotion_df['emotion'], emotion_df['genre'])
        if crosstab.shape[0] > 10:
            crosstab = crosstab.head(10)
        if crosstab.shape[1] > 15:
            crosstab = crosstab.iloc[:, :15]
        plt.figure(figsize=(12, 8))
        crosstab.plot(kind='bar', stacked=True)
        plt.title('감정별 장르 선호도', fontsize=14)
        plt.xlabel('감정')
        plt.ylabel('선택 횟수')
        plt.xticks(rotation=45)
        plt.legend(title='장르', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()
        return plot_data
    except Exception as e:
        logger.error(f"Emotion plot error: {e}")
        return ""

def create_face_plot(df):
    try:
        # 한글 폰트 설정 (NanumGothic 우선 적용)
        import matplotlib
        import matplotlib.font_manager as fm
        available_fonts = {f.name for f in fm.fontManager.ttflist}
        if 'NanumGothic' in available_fonts:
            matplotlib.rcParams['font.family'] = 'NanumGothic'
        elif 'Malgun Gothic' in available_fonts:
            matplotlib.rcParams['font.family'] = 'Malgun Gothic'
        elif 'AppleGothic' in available_fonts:
            matplotlib.rcParams['font.family'] = 'AppleGothic'
        else:
            matplotlib.rcParams['font.family'] = 'DejaVu Sans'
        matplotlib.rcParams['axes.unicode_minus'] = False

        face_data = df[df['face_shape'] != 'Unknown']
        if face_data.empty:
            return ""
        face_genre_data = []
        for _, row in face_data.iterrows():
            if (pd.notna(row['face_shape']) and pd.notna(row['genres']) and
                len(str(row['genres'])) < 1000):
                genres = [g.strip() for g in str(row['genres']).split(',') if len(g.strip()) < 100]
                for genre in genres[:10]:
                    face_genre_data.append({
                        'face_shape': str(row['face_shape'])[:50],
                        'genre': genre
                    })
        if not face_genre_data:
            return None  # 데이터가 없으면 None 반환
        face_df = pd.DataFrame(face_genre_data)
        crosstab = pd.crosstab(face_df['face_shape'], face_df['genre'])
        plt.figure(figsize=(12, 8))
        crosstab.plot(kind='bar', stacked=True)
        plt.title('얼굴형별 장르 선호도', fontsize=14)
        plt.xlabel('얼굴형')
        plt.ylabel('선택 횟수')
        plt.xticks(rotation=45)
        plt.legend(title='장르', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()
        return plot_data
    except Exception as e:
        logger.error(f"Face plot error: {e}")
        return ""

@app.route('/recommend')
def recommend():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM analysis_results", conn)
        conn.close()
        if df.empty:
            return render_template('recommend.html', 
                                recommendations=[], 
                                insights=[], 
                                total_users=0)
        recommendations = generate_recommendations(df)
        insights = generate_insights(df)
        return render_template('recommend.html',
                            recommendations=recommendations,
                            insights=insights,
                            total_users=len(df))
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return render_template('recommend.html', 
                            recommendations=[], 
                            insights=[], 
                            total_users=0)

def generate_recommendations(df):
    try:
        recommendations = []
        for emotion in df['emotion'].unique():
            if pd.isna(emotion) or len(str(emotion)) > 50:
                continue
            emotion_data = df[df['emotion'] == emotion]
            genre_counts = {}
            for genres_str in emotion_data['genres'].dropna():
                if genres_str and len(str(genres_str)) < 1000:
                    genres = [g.strip() for g in str(genres_str).split(',') if len(g.strip()) < 100]
                    for genre in genres[:10]:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
            if genre_counts:
                top_genre = max(genre_counts, key=genre_counts.get)
                count = genre_counts[top_genre]
                percentage = (count / len(emotion_data)) * 100
                recommendations.append({
                    'emotion': str(emotion),
                    'genre': str(top_genre),
                    'count': count,
                    'percentage': round(percentage, 1)
                })
        return sorted(recommendations, key=lambda x: x['percentage'], reverse=True)[:20]
    except Exception as e:
        logger.error(f"Recommendation generation error: {e}")
        return []

def generate_insights(df):
    try:
        insights = []
        total_users = len(df)
        if total_users == 0:
            return ["아직 분석된 데이터가 없습니다."]
        insights.append(f"총 {total_users:,}명의 사용자가 분석을 완료했습니다.")
        if not df['emotion'].isna().all():
            emotion_counts = df['emotion'].value_counts()
            if len(emotion_counts) > 0:
                most_common_emotion = emotion_counts.index[0]
                emotion_count = emotion_counts.iloc[0]
                insights.append(f"가장 많이 감지된 감정은 '{most_common_emotion}'입니다. ({emotion_count:,}명)")
        if not df['age'].isna().all():
            valid_ages = df['age'][(df['age'] >= 0) & (df['age'] <= 150)]
            if len(valid_ages) > 0:
                avg_age = valid_ages.mean()
                insights.append(f"사용자 평균 나이는 {avg_age:.1f}세입니다.")
        if not df['gender'].isna().all():
            gender_counts = df['gender'].value_counts()
            if len(gender_counts) > 0:
                top_gender = gender_counts.index[0]
                gender_percentage = (gender_counts.iloc[0] / total_users) * 100
                insights.append(f"사용자의 {gender_percentage:.1f}%가 {top_gender}으로 분석되었습니다.")
        # 얼굴형 인사이트 추가
        if 'face_shape' in df.columns and not df['face_shape'].isna().all():
            face_shape_counts = df['face_shape'].value_counts()
            if len(face_shape_counts) > 0 and face_shape_counts.index[0] != "Unknown":
                most_common_face_shape = face_shape_counts.index[0]
                face_shape_count = face_shape_counts.iloc[0]
                insights.append(
                    f"가장 많이 감지된 얼굴형은 '{most_common_face_shape}'입니다. ({face_shape_count:,}명)"
                )
        # 장르 인사이트 추가
        genre_series = df['genres'].dropna().apply(
            lambda x: [g.strip() for g in str(x).split(',') if g.strip()]
        )
        all_genres = [g for genres in genre_series for g in genres]
        if all_genres:
            top_genre = pd.Series(all_genres).value_counts().index[0]
            top_genre_count = pd.Series(all_genres).value_counts().iloc[0]
            insights.append(f"가장 인기 있는 장르는 '{top_genre}'입니다. ({top_genre_count:,}회 선택)")
        return insights
    except Exception as e:
        logger.error(f"Insight generation error: {e}")
        return ["데이터 인사이트를 생성하는 중 오류가 발생했습니다."]

# --- 관리자 인증 데코레이터 ---
def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('is_admin'):
            flash('관리자 로그인 후 이용 가능합니다.', 'error')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

# --- 관리자 로그인 라우트 ---
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        password = request.form.get('password', '')
        if check_password_hash(ADMIN_PASSWORD_HASH, password):
            session['is_admin'] = True
            flash('관리자 로그인 성공', 'success')
            return redirect(url_for('admin'))
        else:
            flash('비밀번호가 올바르지 않습니다.', 'error')
    return render_template('admin_login.html')

@app.route('/admin_logout')
def admin_logout():
    session.pop('is_admin', None)
    flash('로그아웃되었습니다.', 'info')
    return redirect(url_for('index'))

# --- app_extra_routes.py의 라우트 등록 (Flask blueprint 방식이 아니면 직접 import) ---
import app_extra_routes

# --- 관리자 전용 라우트에 인증 적용 (import 이후에 적용!) ---
for func_name in ['correlation', 'cluster_view', 'train_mlp']:
    if func_name in app.view_functions:
        app.view_functions[func_name] = admin_required(app.view_functions[func_name])

# download_csv는 직접 데코레이터 적용
@app.route('/download_csv')
@admin_required
def download_csv():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM analysis_results", conn)
        conn.close()
        if df.empty:
            flash('다운로드할 데이터가 없습니다.', 'warning')
            return redirect(url_for('index'))
        csv_io = BytesIO()
        df.to_csv(csv_io, index=False, encoding='utf-8-sig')
        csv_io.seek(0)
        return send_file(
            csv_io,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"analysis_results_{datetime.now().strftime('%Y%m%d')}.csv"
        )
    except Exception as e:
        logger.error(f"CSV 다운로드 실패: {e}")
        flash('CSV 다운로드 중 오류가 발생했습니다.', 'error')
        return redirect(url_for('index'))

@app.route('/graph')
def graph():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM analysis_results", conn)
        conn.close()
        if df.empty:
            flash('아직 데이터가 없습니다. 이미지를 업로드해 분석을 먼저 진행해주세요.', 'warning')
            return render_template('graph.html', genre_plot='', emotion_plot='', face_plot='')
        genre_plot = create_genre_plot(df)
        emotion_plot = create_emotion_plot(df)
        face_plot = create_face_plot(df)
        return render_template('graph.html', genre_plot=genre_plot, emotion_plot=emotion_plot, face_plot=face_plot)
    except Exception as e:
        logger.error(f"Graph page error: {e}")
        flash('그래프를 생성하는 중 오류가 발생했습니다.', 'error')
        return render_template('graph.html', genre_plot='', emotion_plot='', face_plot='')

@app.route('/admin')
def admin():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM analysis_results ORDER BY id DESC LIMIT 100", conn)
        conn.close()
        if df.empty:
            table_html = '<div style="color:#888; font-size:1.1em; margin:20px 0;">아직 데이터가 없습니다.</div>'
        else:
            # 주요 컬럼만 표시, 너무 길면 자름
            show_cols = ['timestamp', 'age', 'gender', 'emotion', 'genres', 'face_shape']
            for col in show_cols:
                if col not in df.columns:
                    df[col] = ''
            table_html = df[show_cols].to_html(classes='data-table', index=False, border=0, justify='center')
        return render_template('admin.html', table_html=table_html)
    except Exception as e:
        return render_template('admin.html', table_html=f'<div style="color:red">오류: {e}</div>')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/upload', methods=['GET'])
def upload_get():
    # GET 요청은 index로 리다이렉트
    flash('이미지 업로드는 메인 화면에서 진행해주세요.', 'info')
    return redirect(url_for('index'))

@app.route('/project_info')
def project_info():
    """프로젝트 목적 및 전체 과정 요약 페이지 (논문/보고서/발표용)"""
    info = {
        "목적": [
            "얼굴 이미지 기반의 영화 취향 예측 (표정, 얼굴형, 피부톤 등 AI 분석)",
            "개인 맞춤형 추천 시스템 구현 (기존 나이·성별 기반보다 정교함)",
            "감정/외형 정보와 콘텐츠 선호도의 상관관계 분석 및 시각화"
        ],
        "개발과정": [
            "데이터 수집 및 전처리: 얼굴 이미지, DeepFace/CNN 분석, DB 저장",
            f"AI 모델 개발: 얼굴형 분류 CNN (버전 {FACE_SHAPE_MODEL_VERSION}), DeepFace 감정 분석, 패턴 학습",
            "Flask 웹앱: 업로드, 분석, 추천, 관리자, 통계, 시각화 등 구현",
            "기능: 자동 얼굴 분석, 결과 시각화, 관리자/통계/상관관계/클러스터링 등"
        ],
        "출력/배포": [
            "전체 ZIP 패키징, 로컬/클라우드 배포, 즉시 실행 가능"
        ],
        "활용성": [
            "AI 기반 개인화 추천 연구, 마케팅, OTT 서비스, GPT 연동 확장성"
        ],
        "연동": [
            f"얼굴형 분류 모델(버전 {FACE_SHAPE_MODEL_VERSION}) 및 DeepFace 분석 결과가 Flask와 실시간 연동되어 DB/추천/시각화에 반영됨"
        ]
    }
    return render_template('project_info.html', info=info, model_version=FACE_SHAPE_MODEL_VERSION)

@app.route('/upload', methods=['POST'])
def upload():
    if request.method != 'POST':
        return redirect(url_for('index'))
    if 'file' not in request.files:
        flash('파일이 첨부되지 않았습니다.', 'error')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('파일명이 비어 있습니다.', 'error')
        return redirect(url_for('index'))
    if not allowed_file(file.filename):
        flash('허용되지 않는 파일 형식입니다.', 'error')
        return redirect(url_for('index'))
    file_content = file.read()
    if not validate_mime_type(file_content):
        flash('이미지 파일이 아닙니다.', 'error')
        return redirect(url_for('index'))
    file_hash = get_file_hash(file_content)
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{file_hash}.{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(save_path, 'wb') as f:
        f.write(file_content)
    # 얼굴 분석
    try:
        face_result = analyze_face(save_path)
        if face_shape_model is None:
            face_shape = "Unknown"
            face_shape_prob = 0.0
            flash('⚠️ 얼굴형 분류 모델이 준비되지 않아 얼굴형 분석 결과는 제공되지 않습니다.', 'warning')
        else:
            face_shape, face_shape_prob = predict_face_shape(save_path)
        embedding = get_face_embedding(save_path)
    except Exception as e:
        logger.error(f"분석 실패: {e}")
        flash('얼굴 분석에 실패했습니다.', 'error')
        return redirect(url_for('index'))
    # DB 저장
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO analysis_results (
                timestamp, age, gender, gender_confidence, emotion, emotion_confidence, emotion_scores, genres, filename_hash, face_shape, embedding, ip_address, user_agent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            face_result['age'],
            face_result['gender'],
            face_result['gender_confidence'],
            face_result['emotion'],
            face_result['emotion_confidence'],
            json.dumps(face_result['emotion_scores'], ensure_ascii=False),
            '',  # 추천 장르(선택 전)
            file_hash,
            face_shape,
            embedding,
            get_client_ip(),
            sanitize_user_agent()
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB 저장 실패: {e}")
        flash('DB 저장에 실패했습니다.', 'error')
        return redirect(url_for('index'))
    # 결과 페이지 렌더링
    return render_template('result.html', 
        age=face_result['age'],
        gender=face_result['gender'],
        gender_confidence=face_result['gender_confidence'],
        emotion=face_result['emotion'],
        emotion_confidence=face_result['emotion_confidence'],
        emotion_scores=face_result['emotion_scores'],
        face_shape=face_shape,
        face_shape_prob=face_shape_prob,
        filename=filename,
        model_version=FACE_SHAPE_MODEL_VERSION
    )

if __name__ == "__main__":
    # 개발/테스트 환경에서는 0.0.0.0:5000으로 실행 (외부 접속 허용)
    app.run(host="0.0.0.0", port=5000, debug=True)
