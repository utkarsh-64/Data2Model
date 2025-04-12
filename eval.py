import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix
)

# Load model function
def load_model(file):
    """ Load the model from a pickle file """
    try:
        model = pickle.load(file)
        if not hasattr(model, "predict"):
            raise TypeError("The loaded object is not a valid model. Ensure the correct model is uploaded.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Regression Evaluation
def plot_regression_evaluation(mae, mse, rmse, r2):
    """ Visualize regression metrics on a scale """
    fig, ax = plt.subplots(figsize=(9, 2))

    max_value = max(mae, mse, rmse) * 1.5  
    scale_values = np.linspace(0, max_value, 6)
    colors = ["#2ECC71", "#F1C40F", "#F39C12", "#E67E22", "#C0392B"]

    for i in range(len(scale_values) - 1):
        ax.barh(y=0, width=scale_values[i+1] - scale_values[i], left=scale_values[i],
                color=colors[i], height=0.3, edgecolor="black", alpha=0.8)

    # Plot points for each metric
    ax.scatter(mae, 0, color="#3498DB", s=90, edgecolors="black", linewidth=1.2, label=f"MAE: {mae:.2f}", zorder=3)
    ax.scatter(mse, 0, color="#9B59B6", s=90, edgecolors="black", linewidth=1.2, label=f"MSE: {mse:.2f}", zorder=3)
    ax.scatter(rmse, 0, color="#34495E", s=100, edgecolors="black", linewidth=1.5, label=f"RMSE: {rmse:.2f}", zorder=3)
    ax.scatter(r2 * max_value, 0, color="#E74C3C", s=90, edgecolors="black", linewidth=1.2, label=f"R²: {r2:.2f}", zorder=3)

    ax.set_xlim(0, max_value)
    ax.set_yticks([])
    ax.set_xticks(scale_values)
    ax.set_xticklabels([f"{int(val)}" for val in scale_values], fontsize=10, fontweight="bold", color="#2C3E50")
    ax.set_title("📊 Model Evaluation Scale", fontsize=13, fontweight="bold", color="#2C3E50")

    legend = plt.legend(fontsize=10, loc="upper right", frameon=True, edgecolor="black", facecolor="white")
    for text in legend.get_texts():
        text.set_color("#2C3E50")

    plt.grid(False)
    st.pyplot(fig)

def evaluate_regression(model, X_test, y_test):
    """ Evaluate regression model with visual metrics """
    try:
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        st.subheader("Regression Model Evaluation")
        st.json({
            "Mean Absolute Error": mae,
            "Mean Squared Error": mse,
            "Root Mean Squared Error": rmse,
            "R2 Score": r2
        })

        plot_regression_evaluation(mae, mse, rmse, r2)
    except Exception as e:
        st.error(f"Error during regression evaluation: {e}")

# Classification Evaluation
# Classification Evaluation
def evaluate_classification(model, X_test, y_test):
    """ Evaluate a classification model with a table format and an improved Confusion Matrix visualization """
    try:
        predictions = model.predict(X_test)

        st.subheader("Classification Report")
        report = classification_report(y_test, predictions, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report.style.format(precision=2))  # Display as a styled table
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, predictions)
        labels = sorted(set(y_test))

        fig, ax = plt.subplots(figsize=(4.5, 3.5))  # Adjusted for better visibility
        sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=labels, 
                    yticklabels=labels, linewidths=0.5, cbar=False, annot_kws={"size": 10})
        plt.xlabel("Predicted Labels", fontsize=10)
        plt.ylabel("True Labels", fontsize=10)
        plt.title("Confusion Matrix", fontsize=11)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during classification evaluation: {e}")


# Data Preprocessing
def preprocess_data(df, target_column):
    """ Convert categorical columns to numerical, ensure consistent column names, and handle missing values """
    try:
        df = df.dropna()
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype("category").cat.codes  
        df.columns = df.columns.astype(str)
        X_test = df.drop(columns=[target_column])
        y_test = df[target_column]
        return X_test, y_test
    except Exception as e:
        st.error(f"Error in data preprocessing: {e}")
        return None, None

# Main Function
def main():
    st.title("📊 Model Evaluation Dashboard")
    
    model_file = st.file_uploader("Upload your trained model (.pkl)", type=["pkl"])
    data_file = st.file_uploader("Upload test data (CSV)", type=["csv"])
    
    task = st.radio("Select Model Type", ("Regression", "Classification"))
    
    if model_file and data_file:
        model = load_model(model_file)
        df = pd.read_csv(data_file)
        
        if model is not None and df is not None:
            target_column = st.selectbox("Select Target Column", df.columns)
            
            if target_column:
                X_test, y_test = preprocess_data(df, target_column)
                
                if X_test is not None and y_test is not None:
                    if task == "Regression":
                        evaluate_regression(model, X_test, y_test)
                    else:
                        st.subheader("Classification Model Evaluation")
                        evaluate_classification(model, X_test, y_test)

if __name__ == "__main__":
    main()