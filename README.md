# 얼굴 분석 및 영화 장르 추천 웹앱

이 프로젝트는 업로드된 얼굴 사진을 분석하여 나이, 성별, 감정 등을 추론하고
사용자가 선택한 영화 장르 정보를 함께 저장합니다. 저장된 데이터는 통계
그래프와 추천 기능을 통해 확인할 수 있습니다.

## 실행 방법

1. **Python 3.10 환경을 권장합니다.**
   Python 3.12를 사용할 경우 DeepFace가 PyPI에서 정상 설치되지 않을 수 있습니다.
   아래 명령으로 GitHub 저장소에서 직접 설치할 수 있습니다.
   ```bash
   pip install git+https://github.com/serengil/deepface.git
   ```
2. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```
3. 데이터베이스 초기화 및 서버 실행:
   ```bash
   python app.py
   ```
4. 브라우저에서 `http://localhost:5000` 으로 접속하여 서비스를 이용합니다.

`models/class_indices.json` 파일이 없으면 기본 라벨을 사용합니다.
DeepFace가 설치되어 있지 않으면 업로드 시 오류 메시지가 표시됩니다.
