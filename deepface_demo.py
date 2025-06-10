import os
import argparse

from deepface import DeepFace


def main() -> None:
    """Analyze a local image with DeepFace using offline weights."""
    parser = argparse.ArgumentParser(description="Analyze an image with DeepFace")
    parser.add_argument(
        "image",
        help="Path to the image file to analyze",
    )
    parser.add_argument(
        "--weights-dir",
        dest="weights_dir",
        default=None,
        help=(
            "Directory containing pre-downloaded DeepFace weights. "
            "If set, the path will be used as DEEPFACE_HOME."
        ),
    )
    args = parser.parse_args()

    if args.weights_dir:
        os.environ["DEEPFACE_HOME"] = os.path.abspath(args.weights_dir)

    img_path = args.image
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    print("Analyzing image with DeepFace...")
    result = DeepFace.analyze(
        img_path=img_path,
        actions=["age", "gender", "emotion"],
        enforce_detection=False,
    )
    print(result)


if __name__ == "__main__":
    main()
