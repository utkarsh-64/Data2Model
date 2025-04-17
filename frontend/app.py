from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import pickle
import io
import os
import uuid
import secrets
import zipfile
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, KBinsDiscretizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
CORS(app, supports_credentials=True)

UPLOAD_DIR = "tmp"
EXPORT_DIR = "exports"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
plt.switch_backend("Agg")

def load_df(df_id):
    path = os.path.join(UPLOAD_DIR, f"{df_id}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("Dataset not found")
    return pd.read_pickle(path)

@app.route("/")
def home():
    return "Backend Running Successfully"

@app.route("/upload-data", methods=["POST"])
def upload_data():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        df = pd.read_csv(file) if file.filename.endswith(".csv") else pd.read_excel(file)
        df_id = str(uuid.uuid4())
        df.to_pickle(os.path.join(UPLOAD_DIR, f"{df_id}.pkl"))
        return jsonify({
            "message": "File uploaded successfully",
            "df_id": df_id,
            "preview": df.head(5).to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/eda", methods=["POST"])
def eda():
    try:
        df_id = request.json.get("df_id")
        df = load_df(df_id)
        summary = {
            "total_records": df.shape[0],
            "total_features": df.shape[1],
            "numeric_count": df.select_dtypes(include="number").shape[1],
            "categorical_count": df.select_dtypes(exclude="number").shape[1]
        }
        columns = []
        for col in df.columns:
            columns.append({
                "name": col,
                "type": "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical",
                "missing": df[col].isnull().mean(),
                "unique": df[col].nunique(),
                "sample_values": [str(v) for v in df[col].dropna().unique()[:5]]
            })
        return jsonify({"summary": summary, "columns": columns})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/clean-data", methods=["POST"])
def clean_data():
    try:
        data = request.get_json()
        df = load_df(data["df_id"])
        column = data["column"]
        method = data["method"]
        custom_value = data.get("customValue")

        if method == "mean":
            value = df[column].mean()
        elif method == "median":
            value = df[column].median()
        elif method == "mode":
            value = df[column].mode()[0]
        elif method == "custom":
            value = df[column].dtype.type(custom_value)
        else:
            return jsonify({"error": "Invalid method"}), 400

        df[column].fillna(value, inplace=True)
        df.to_pickle(os.path.join(UPLOAD_DIR, f"{data['df_id']}.pkl"))
        return jsonify({"message": f"{method} imputation applied to {column}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/drop-columns", methods=["POST"])
def drop_columns():
    try:
        data = request.get_json()
        df = load_df(data["df_id"])
        df.drop(columns=data["columns"], inplace=True)
        df.to_pickle(os.path.join(UPLOAD_DIR, f"{data['df_id']}.pkl"))
        return jsonify({"message": f"Dropped columns: {data['columns']}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/scale", methods=["POST"])
def scale():
    try:
        data = request.get_json()
        df = load_df(data["df_id"])
        method = data["method"]
        columns = data["columns"]

        if method == "minmax":
            scaler = MinMaxScaler()
        elif method == "standard":
            scaler = StandardScaler()
        elif method == "robust":
            scaler = RobustScaler()
        elif method == "log":
            df[columns] = np.log1p(df[columns])
            df.to_pickle(os.path.join(UPLOAD_DIR, f"{data['df_id']}.pkl"))
            return jsonify({"message": f"Applied log scaling to {columns}"})
        else:
            return jsonify({"error": "Invalid scaling method"}), 400

        df[columns] = scaler.fit_transform(df[columns])
        df.to_pickle(os.path.join(UPLOAD_DIR, f"{data['df_id']}.pkl"))
        return jsonify({"message": f"{method} scaling applied"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train-model", methods=["POST"])
def train_model():
    try:
        data = request.get_json()
        df = load_df(data["df_id"])
        target = data["target"]
        model_type = data["model_type"]
        test_size = float(data.get("test_size", 0.2))
        scale = data.get("scale", True)
        algorithms = data["algorithms"]

        y = df[target]
        X = pd.get_dummies(df.drop(columns=[target]))

        if scale:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        metrics = {}
        models = {}
        for algo in algorithms:
            if model_type == "classification":
                model_map = {
                    "knn": KNeighborsClassifier(),
                    "random_forest": RandomForestClassifier(),
                    "logistic_regression": LogisticRegression(max_iter=1000),
                    "decision_trees": DecisionTreeClassifier(),
                    "naive_bayes": GaussianNB(),
                    "sgd": SGDClassifier(),
                    "svm": SVC(probability=True)
                }
            else:
                model_map = {
                    "knn": KNeighborsRegressor(),
                    "linear": LinearRegression(),
                    "ridge": Ridge(),
                    "lasso": Lasso(),
                    "decision_trees": DecisionTreeRegressor(),
                    "random_forest": RandomForestRegressor(),
                    "svm": SVR()
                }

            model = model_map.get(algo)
            if model is None:
                metrics[algo] = {"error": "Unsupported algorithm"}
                continue

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            models[f"{algo}_model.pkl"] = pickle.dumps(model)

            if model_type == "classification":
                metrics[algo] = {
                    "accuracy": round(accuracy_score(y_test, y_pred), 3),
                    "f1": round(f1_score(y_test, y_pred, average='weighted'), 3),
                    "precision": round(precision_score(y_test, y_pred, average='weighted'), 3),
                    "recall": round(recall_score(y_test, y_pred, average='weighted'), 3)
                }
            else:
                metrics[algo] = {
                    "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
                    "mae": round(mean_absolute_error(y_test, y_pred), 3),
                    "r2": round(r2_score(y_test, y_pred), 3)
                }

        zip_token = secrets.token_hex(8)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        zip_filename = f"trained_models_{timestamp}_{zip_token}.zip"
        zip_path = os.path.join(EXPORT_DIR, zip_filename)

        with zipfile.ZipFile(zip_path, "w") as zf:
            for filename, content in models.items():
                zf.writestr(filename, content)
            zf.writestr("train_data.csv", pd.concat([X_train, y_train], axis=1).to_csv(index=False))
            zf.writestr("test_data.csv", pd.concat([X_test, y_test], axis=1).to_csv(index=False))
            if scale:
                zf.writestr("scaler.pkl", pickle.dumps(scaler))

        return jsonify({
            "message": "Training complete",
            "metrics": metrics,
            "zip_path": f"/download-model/{zip_filename}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download-model/<zip_filename>")
def download_model(zip_filename):
    zip_path = os.path.join(EXPORT_DIR, zip_filename)
    if not os.path.exists(zip_path):
        return "File not found", 404
    return send_file(zip_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
