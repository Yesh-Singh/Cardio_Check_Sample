from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import mimetypes
import os
import threading
from urllib.parse import unquote, urlparse

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "cardio_train.csv"
MODEL_LOCK = threading.Lock()


def train_models():
    data = pd.read_csv(DATA_FILE, sep=";")
    data = data.drop(columns=["id"], errors="ignore").drop_duplicates()
    data["age"] = (data["age"] / 365.25).round().astype(int)
    data["bmi"] = data["weight"] / ((data["height"] / 100) ** 2)
    data = data[(data["ap_lo"] >= 40) & (data["ap_lo"] <= 180)]
    data = data[(data["ap_hi"] >= 70) & (data["ap_hi"] <= 220)]
    data = data[data["ap_hi"] >= data["ap_lo"]]
    data = data[(data["height"] >= 100) & (data["height"] <= 220)]
    data = data[(data["weight"] >= 30) & (data["weight"] <= 200)]

    features = data.drop(columns=["cardio"])
    target = data["cardio"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear SVM": LinearSVC(C=1.0, max_iter=2000, random_state=42),
    }
    trained = {}
    for name, model in models.items():
        model.fit(x_train_scaled, y_train)
        trained[name] = (model, accuracy_score(y_test, model.predict(x_test_scaled)))
    best_name = max(trained, key=lambda name: trained[name][1])
    return features.columns.tolist(), scaler, trained[best_name][0], best_name, trained[best_name][1]


FEATURE_COLUMNS, SCALER, MODEL, MODEL_NAME, MODEL_ACCURACY = train_models()


def classify_level(decision_score):
    if decision_score >= 0.8:
        return "high"
    if decision_score >= -0.8:
        return "medium"
    return "low"


def make_prediction(patient):
    patient["bmi"] = patient["weight"] / ((patient["height"] / 100) ** 2)
    row = pd.DataFrame([patient])[FEATURE_COLUMNS]
    scaled = SCALER.transform(row)
    prediction = int(MODEL.predict(scaled)[0])
    decision_score = float(MODEL.decision_function(scaled)[0])
    factors = []
    if patient["age"] >= 50:
        factors.append("Age 50+")
    if patient["bmi"] >= 30:
        factors.append("BMI in obesity range")
    elif patient["bmi"] >= 25:
        factors.append("BMI above healthy range")
    if patient["ap_hi"] >= 140 or patient["ap_lo"] >= 90:
        factors.append("High blood pressure")
    elif patient["ap_hi"] >= 130 or patient["ap_lo"] >= 80:
        factors.append("Elevated blood pressure")
    if patient["cholesterol"] > 1:
        factors.append("Cholesterol above normal")
    if patient["gluc"] > 1:
        factors.append("Glucose above normal")
    if patient["smoke"]:
        factors.append("Current smoking")
    if not patient["active"]:
        factors.append("Low physical activity")
    return {
        "prediction": prediction,
        "level": classify_level(decision_score),
        "modelScore": round(decision_score, 3),
        "model": MODEL_NAME,
        "accuracy": round(MODEL_ACCURACY * 100, 2),
        "bmi": round(patient["bmi"], 1),
        "factors": factors,
    }


class CardioHandler(BaseHTTPRequestHandler):
    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json({"status": "ok", "model": MODEL_NAME, "accuracy": round(MODEL_ACCURACY * 100, 2)})
            return
        requested_path = unquote(urlparse(self.path).path)
        if requested_path == "/":
            requested_path = "/index.html"
        asset = (ROOT / requested_path.lstrip("/")).resolve()
        try:
            asset.relative_to(ROOT.resolve())
        except ValueError:
            self.send_json({"error": "Not found"}, 404)
            return
        if asset.is_file():
            body = asset.read_bytes()
            content_type = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path != "/predict":
            self.send_json({"error": "Not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            patient = json.loads(self.rfile.read(length))
            required = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
            if any(field not in patient for field in required):
                raise ValueError("Missing patient field")
            normalized = {field: float(patient[field]) for field in required}
            normalized["age"] = int(normalized["age"])
            normalized["gender"] = int(normalized["gender"])
            normalized["cholesterol"] = int(normalized["cholesterol"])
            normalized["gluc"] = int(normalized["gluc"])
            normalized["smoke"] = int(normalized["smoke"])
            normalized["alco"] = int(normalized["alco"])
            normalized["active"] = int(normalized["active"])
            self.send_json(make_prediction(normalized))
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, 400)


if __name__ == "__main__":
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), CardioHandler)
    print(f"CardioCheck running on port {port} using {MODEL_NAME} ({MODEL_ACCURACY * 100:.2f}% test accuracy)")
    server.serve_forever()
