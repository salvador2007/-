import os
import requests
from deepface import DeepFace


def download_image(url, filename):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    with open(filename, 'wb') as f:
        f.write(resp.content)


def main():
    img_url = "https://raw.githubusercontent.com/serengil/deepface/master/tests/dataset/img1.jpg"
    local_path = "sample.jpg"
    if not os.path.exists(local_path):
        print("Downloading sample image...")
        download_image(img_url, local_path)
    print("Analyzing image with DeepFace...")
    result = DeepFace.analyze(img_path=local_path, actions=["age", "gender", "emotion"], enforce_detection=False)
    print(result)


if __name__ == "__main__":
    main()
