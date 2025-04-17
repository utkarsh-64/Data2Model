from flask import Flask, request, jsonify, send_file , session
from flask_cors import CORS
import pandas as pd
import pickle
import io
from io import BytesIO
import secrets
from datetime import datetime
import zipfile
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, KBinsDiscretizer, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os, uuid


app = Flask(__name__)
CORS(app,supports_credentials=True)
app.secret_key = '14454af5f5c8e90e1e90'  # Required for session
UPLOAD_DIR = "tmp"
os.makedirs(UPLOAD_DIR, exist_ok=True)
EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)
matplotlib.use('Agg')


# Globals
DATA = {}
MODELS = {}
SCALER = None

@app.route('/upload-data', methods=['POST'])
def upload_data():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        # Read DataFrame
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400

        # Save it to a temp file and store key in session
        df_key = str(uuid.uuid4())
        df_path = os.path.join(UPLOAD_DIR, f"{df_key}.pkl")
        df.to_pickle(df_path)
        session['df_key'] = df_key

        preview = df.head(5).to_dict(orient='records')
        return jsonify({
            'message': 'File uploaded successfully',
            'rows': df.shape[0],
            'columns': df.shape[1],
            'preview': preview
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/eda', methods=['POST'])
def eda():
    df_key = session.get('df_key')
    if not df_key:
        return jsonify({'error': 'No uploaded dataset found'}), 400

    try:
        df_path = os.path.join(UPLOAD_DIR, f"{df_key}.pkl")
        df = pd.read_pickle(df_path)

        summary = {
            'total_records': df.shape[0],
            'total_features': df.shape[1],
            'numeric_count': df.select_dtypes(include='number').shape[1],
            'categorical_count': df.select_dtypes(exclude='number').shape[1],
        }

        columns = []
        for col in df.columns:
            values = df[col].dropna().unique()[:5]
            columns.append({
                'name': col,
                'type': 'numeric' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical',
                'missing': df[col].isnull().mean(),
                'unique': df[col].nunique(),
                'sample_values': [str(v) for v in values]
            })

        return jsonify({
            'summary': summary,
            'columns': columns
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def load_df_from_session():
    df_key = session.get("df_key")
    if not df_key:
        raise ValueError("No uploaded dataset found")
    path = os.path.join("tmp", f"{df_key}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("Dataset file not found")
    return pd.read_pickle(path)

@app.route("/eda/distribution", methods=["GET"])
def eda_distribution():
    try:
        df = load_df_from_session()
        numeric_cols = df.select_dtypes(include="number").columns[:6]  # show up to 6

        fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(6, 4 * len(numeric_cols)))
        if len(numeric_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, numeric_cols):
            sns.histplot(df[col].dropna(), ax=ax, kde=True, bins=30)
            ax.set_title(f"Distribution of {col}")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/eda/correlation", methods=["GET"])
def eda_correlation():
    try:
        df = load_df_from_session()
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            return jsonify({"error": "Not enough numeric columns for correlation"}), 400

        corr = numeric_df.corr(numeric_only=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")

        buf.seek(0)
        plt.close()
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/eda/missing", methods=["GET"])
def eda_missing():
    try:
        df = load_df_from_session()
        missing_pct = df.isnull().mean()

        fig, ax = plt.subplots(figsize=(8, 4))
        missing_pct[missing_pct > 0].sort_values(ascending=False).plot.bar(ax=ax)
        ax.set_ylabel("Proportion Missing")
        ax.set_title("Missing Values per Column")
        plt.xticks(rotation=45)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/clean-data', methods=['POST'])
def clean_data():
    import numpy as np
    df_key = session.get('df_key')
    if not df_key:
        return jsonify({'error': 'No uploaded dataset found'}), 400

    df_path = os.path.join('tmp', f"{df_key}.pkl")
    if not os.path.exists(df_path):
        return jsonify({'error': 'Dataset file missing'}), 404

    try:
        df = pd.read_pickle(df_path)

        data = request.get_json()
        column = data.get('column')
        method = data.get('method')
        custom_value = data.get('customValue')

        if column not in df.columns:
            return jsonify({'error': f'Column "{column}" not found'}), 400

        if method == 'mean':
            value = df[column].mean()
        elif method == 'median':
            value = df[column].median()
        elif method == 'mode':
            value = df[column].mode()[0] if not df[column].mode().empty else np.nan
        elif method == 'custom':
            try:
                # Attempt to cast to original dtype
                value = df[column].dtype.type(custom_value)
            except:
                value = custom_value
        else:
            return jsonify({'error': f'Unknown method: {method}'}), 400

        df[column].fillna(value, inplace=True)
        df.to_pickle(df_path)

        return jsonify({
            'message': f'{method} imputation applied to {column}',
            'value_used': value if method != 'custom' else custom_value
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/drop-columns', methods=['POST'])
def drop_columns():
    df_key = session.get('df_key')
    if not df_key:
        return jsonify({'error': 'No uploaded dataset found'}), 400

    try:
        df_path = os.path.join('tmp', f'{df_key}.pkl')
        df = pd.read_pickle(df_path)

        data = request.get_json()
        cols = data.get('columns', [])

        missing = [col for col in cols if col not in df.columns]
        if missing:
            return jsonify({'error': f'Invalid columns: {missing}'}), 400

        df.drop(columns=cols, inplace=True)
        df.to_pickle(df_path)

        return jsonify({'message': f'Dropped columns: {cols}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/replace-values', methods=['POST'])
def replace_values():
    df_key = session.get('df_key')
    if not df_key:
        return jsonify({'error': 'No uploaded dataset found'}), 400

    try:
        df_path = os.path.join('tmp', f'{df_key}.pkl')
        df = pd.read_pickle(df_path)

        data = request.get_json()
        col = data.get('column')
        old = data.get('find')
        new = data.get('replace')

        if col not in df.columns:
            return jsonify({'error': f'Column "{col}" not found'}), 400

        df[col] = df[col].replace(old, new)
        df.to_pickle(df_path)

        return jsonify({'message': f'Replaced {old} with {new} in {col}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/create-feature', methods=['POST'])
def create_feature():
    df_key = session.get('df_key')
    if not df_key:
        return jsonify({'error': 'No uploaded dataset found'}), 400

    try:
        df_path = os.path.join(UPLOAD_DIR, f"{df_key}.pkl")
        df = pd.read_pickle(df_path)

        data = request.get_json()
        new_col = data.get("column")
        expression = data.get("expression")

        df[new_col] = df.eval(expression)
        df.to_pickle(df_path)

        return jsonify({"message": f"Feature '{new_col}' created successfully"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/scale', methods=['POST'])
def scale():
    df_key = session.get('df_key')
    if not df_key:
        return jsonify({'error': 'No uploaded dataset found'}), 400

    try:
        df_path = os.path.join(UPLOAD_DIR, f"{df_key}.pkl")
        df = pd.read_pickle(df_path)

        data = request.get_json()
        method = data.get("method")
        columns = data.get("columns", [])

        if method == "minmax":
            scaler = MinMaxScaler()
            df[columns] = scaler.fit_transform(df[columns])
        elif method == "standard":
            scaler = StandardScaler()
            df[columns] = scaler.fit_transform(df[columns])
        elif method == "robust":
            scaler = RobustScaler()
            df[columns] = scaler.fit_transform(df[columns])
        elif method == "log":
            df[columns] = np.log1p(df[columns])
        else:
            return jsonify({'error': 'Unknown scaling method'}), 400

        df.to_pickle(df_path)
        return jsonify({'message': f'Scaling applied using {method}'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/encode', methods=['POST'])
def encode():
    df_key = session.get('df_key')
    if not df_key:
        return jsonify({'error': 'No uploaded dataset found'}), 400

    try:
        df_path = os.path.join(UPLOAD_DIR, f"{df_key}.pkl")
        df = pd.read_pickle(df_path)

        data = request.get_json()
        method = data.get("method")
        columns = data.get("columns", [])

        if method == "onehot":
            df = pd.get_dummies(df, columns=columns)
        elif method == "label":
            le = LabelEncoder()
            for col in columns:
                df[col] = le.fit_transform(df[col])
        elif method == "target":
            if "target" not in df.columns:
                return jsonify({'error': 'Target column not found in dataset'}), 400
            for col in columns:
                means = df.groupby(col)['target'].mean()
                df[col] = df[col].map(means)
        else:
            return jsonify({'error': 'Unknown encoding method'}), 400

        df.to_pickle(df_path)
        return jsonify({'message': f'{method} encoding applied successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/bin', methods=['POST'])
def bin_column():
    df_key = session.get('df_key')
    if not df_key:
        return jsonify({'error': 'No uploaded dataset found'}), 400

    try:
        df_path = os.path.join(UPLOAD_DIR, f"{df_key}.pkl")
        df = pd.read_pickle(df_path)

        data = request.get_json()
        column = data.get("column")
        bins = int(data.get("bins", 5))
        strategy = data.get("strategy", "uniform")

        discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy)
        df[f"{column}_binned"] = discretizer.fit_transform(df[[column]])

        df.to_pickle(df_path)
        return jsonify({'message': f'{column} binned into {bins} bins using {strategy}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export', methods=['GET'])
def export_dataset():
    try:
        df_key = session.get("df_key")
        print("Session df_key:", df_key)

        if not df_key:
            return jsonify({"error": "No uploaded dataset found"}), 400

        df_path = os.path.join("tmp", f"{df_key}.pkl")
        print("Looking for file at:", df_path)

        if not os.path.exists(df_path):
            return jsonify({"error": "Processed file not found"}), 404

        df = pd.read_pickle(df_path)
        print("Loaded DataFrame shape:", df.shape)

        export_path = os.path.join("tmp", f"{df_key}_export.csv")
        df.to_csv(export_path, index=False)
        print("CSV written to:", export_path)

        return send_file(
            export_path,
            as_attachment=True,
            download_name="cleaned_dataset.csv",
            mimetype="text/csv"
        )
    except Exception as e:
        print("Export Error:", str(e))  # 👈 Super useful
        return jsonify({"error": str(e)}), 500


@app.route("/feature-engineer", methods=["POST"])
def feature_engineer():
    df = DATA.get('cleaned')
    if df is None:
        return jsonify({"error": "No data to engineer"}), 400
    # You can extend this as needed
    DATA['processed'] = df
    return jsonify({"message": "Feature engineering applied"})



@app.route('/train-model', methods=['POST'])
def train_model():
    
    if 'df_key' not in session:
        return jsonify({'error': 'No uploaded dataset found'}), 400

    df_key = session.get("df_key")
    df_path = os.path.join(UPLOAD_DIR, f"{df_key}.pkl")
    if not os.path.exists(df_path):
        return jsonify({'error': 'Dataset file not found'}), 400

    df = pd.read_pickle(df_path)
    if df is None or not isinstance(df, pd.DataFrame):
        return jsonify({'error': 'Invalid or missing dataset in session'}), 400

    data = request.get_json()
    model_type = data.get('model_type', 'classification')
    algorithms = data.get('algorithms', [])
    target = data.get('target')
    test_size = float(data.get('test_size', 0.2))
    cv_folds = int(data.get('cv_folds', 5))  # Not used but accepted
    scale = data.get('scale', True)
    tune_hyperparams = data.get('tune_hyperparams', False)  # Optional

    if not algorithms or not target:
        return jsonify({'error': 'Target column and algorithms are required'}), 400

    if target not in df.columns:
        return jsonify({'error': 'Selected target column not found in dataset'}), 400

    y = df[target]
    X = df.drop(columns=[target])
    X = pd.get_dummies(X)

    if y.isnull().any():
        return jsonify({'error': 'Target column contains missing values'}), 400

    scaler = None
    if scale:
        try:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        except Exception as e:
            return jsonify({'error': f'Scaling failed: {str(e)}'}), 500

    try:
        X_train, X_test, y_train, y = train_test_split(
            X, y, test_size=test_size, random_state=42)
    except Exception as e:
        return jsonify({'error': f'Data split failed: {str(e)}'}), 500

    metrics = {}
    trained_objects = {}  # Store model + optionally scaler

    for algo in algorithms:
        try:
            if model_type == 'classification':
                model = {
                    'knn': KNeighborsClassifier(),
                    'random_forest': RandomForestClassifier(),
                    'logistic_regression': LogisticRegression(max_iter=1000),
                    'decision_trees': DecisionTreeClassifier(),
                    'naive_bayes': GaussianNB(),
                    'sgd': SGDClassifier(),
                    'svm': SVC(probability=True)
                }.get(algo)
            else:
                model = {
                    'knn': KNeighborsRegressor(),
                    'linear': LinearRegression(),
                    'ridge': Ridge(),
                    'lasso': Lasso(),
                    'decision_trees': DecisionTreeRegressor(),
                    'random_forest': RandomForestRegressor(),
                    'svm': SVR()
                }.get(algo)

            if model is None:
                metrics[algo] = {'error': f"Unsupported algorithm '{algo}'"}
                continue

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            trained_objects[f"{algo}_model.pkl"] = pickle.dumps(model)

            if model_type == 'classification':
                metrics[algo] = {
                    'accuracy': round(accuracy_score(y, y_pred), 3),
                    'f1': round(f1_score(y, y_pred, average='weighted'), 3),
                    'precision': round(precision_score(y, y_pred, average='weighted'), 3),
                    'recall': round(recall_score(y, y_pred, average='weighted'), 3)
                }
            else:
                metrics[algo] = {
                    'rmse': round(np.sqrt(mean_squared_error(y, y_pred)), 3),
                    'mae': round(mean_absolute_error(y, y_pred), 3),
                    'r2': round(r2_score(y, y_pred), 3)
                }

        except Exception as e:
            metrics[algo] = {'error': str(e)}

    # Save train/test CSVs
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y], axis=1)

    # Generate a unique filename
    zip_token = secrets.token_hex(8)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    zip_filename = f"trained_models_{timestamp}_{zip_token}.zip"
    zip_path = os.path.join(EXPORT_DIR, zip_filename)

    with zipfile.ZipFile(zip_path, "w") as z:
        for filename, obj in trained_objects.items():
            z.writestr(filename, obj)
        if scale and scaler:
            z.writestr("scaler.pkl", pickle.dumps(scaler))
        z.writestr("train_data.csv", train_df.to_csv(index=False))
        z.writestr("test_data.csv", test_df.to_csv(index=False))

    # Return path instead of storing ZIP in session
    return jsonify({
        "message": "Training completed and ZIP exported successfully",
        "metrics": metrics,
        "zip_path": f"/download-model/{zip_filename}"
    }), 200

    
@app.route('/download-model/<zip_filename>')
def download_model(zip_filename):
    zip_path = os.path.join(EXPORT_DIR, zip_filename)
    if not os.path.exists(zip_path):
        return "File not found", 404
    return send_file(zip_path, as_attachment=True)


@app.route("/evaluate", methods=["POST"])
def evaluate():
    if 'model' not in request.files or 'test_data' not in request.files:
        return jsonify({'error': 'Model and test dataset files are required.'}), 400
    
    model_file = request.files.get("model")
    test_file = request.files.get("test_data")
    task = request.form.get("task","classification")

    if not test_file or test_file.filename == '': 
        return jsonify({'error': 'No test CSV uploaded'}), 400
    
    try: 
        df = pd.read_csv(test_file) 
    except Exception as e: 
        print("Received test file:", test_file.filename)
        return jsonify({'error': f'Failed to read test dataset: {str(e)}'}), 400
    
    try:
        model = pickle.load(model_file)
    except Exception as e:
        return jsonify({'error': f'Failed to load model: {str(e)}'}), 400

    
    # Infer target column (last column)
    if df.shape[1] < 2:
        return jsonify({'error': 'Test dataset must contain at least one feature and one target column.'}), 400

    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

    try:
        y_pred = model.predict(X)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 400

    if task == 'classification':
        report = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)

        return jsonify({
            'task': 'classification',
            'report': report,
            'confusion_matrix': cm.tolist(),  # convert np.array to list for JSON serialization
            'labels': sorted(set(y.tolist()))
        }), 200
    else:  # regression
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        return jsonify({
        'task': 'regression',
        'metrics': {
            'MAE': round(mae, 4),
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'R2': round(r2, 4)
        }
        }), 200
if __name__ == "__main__":
    app.run(debug=True)
