from pathlib import Path

import joblib
import jieba


ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = [
    line.strip()
    for line in (ROOT / "data" / "class.txt").read_text(encoding="utf-8-sig").splitlines()
    if line.strip()
]
VECTORIZER = joblib.load(ROOT / "artifacts" / "random_forest" / "v2" / "tfidf_v2.pkl")
MODEL = joblib.load(ROOT / "artifacts" / "random_forest" / "v2" / "rf_v2.pkl")


def predict(text: str) -> dict[str, object]:
    words = " ".join(jieba.lcut(text)[:20])
    label = int(MODEL.predict(VECTORIZER.transform([words]))[0])
    return {"text": text, "predicted_label": label, "predicted_class_name": CLASS_NAMES[label]}
