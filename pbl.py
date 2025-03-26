import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
import pickle
import numpy as np
import zipfile
import io

def main():
    df = st.file_uploader("Upload The Preprocessed File....", type="csv", accept_multiple_files=False)
    if df:
        df = pd.read_csv(df)
        st.header("Uploaded File")
        st.dataframe(df.head(5))
        target = st.selectbox("Choose the Target Variable", df.columns)
        modeltype = st.selectbox("Choose the Model type", ["Regression", "Classification"])

        if modeltype == "Classification":
            models = st.multiselect("Choose the models you want to select",
                                    ["KNN", "RandomForest", "LogisticRegression", "DecisionTrees", "NaiveBayes", "SGD", "SVM"])
        else:
            models = st.multiselect("Choose the models you want to select",
                                    ["KNN", "Linear", "Ridge", "Lasso", "DecisionTrees", "RandomForest", "SVM"])

        if models:
            st.text(f"You have chosen: {models}")

        but = st.button("Let's Train the model")
        trained_models = {}

        if but:
            X = df.drop(columns=[target])
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Apply scaling if needed
            scaler = None
            if modeltype == "Regression" or any(m in models for m in ["KNN", "SVM", "LogisticRegression", "SGD", "Ridge", "Lasso"]):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            for model_name in models:
                if model_name == "KNN":
                    model = KNeighborsClassifier() if modeltype == "Classification" else KNeighborsRegressor()
                elif model_name == "RandomForest":
                    model = RandomForestClassifier() if modeltype == "Classification" else RandomForestRegressor()
                elif model_name == "LogisticRegression":
                    model = LogisticRegression()
                elif model_name == "DecisionTrees":
                    model = DecisionTreeClassifier() if modeltype == "Classification" else DecisionTreeRegressor()
                elif model_name == "NaiveBayes":
                    model = GaussianNB()
                elif model_name == "SGD":
                    model = SGDClassifier()
                elif model_name == "SVM":
                    model = SVC() if modeltype == "Classification" else SVR()
                elif model_name == "Linear":
                    model = LinearRegression()
                elif model_name == "Ridge":
                    model = Ridge()
                elif model_name == "Lasso":
                    model = Lasso()
                else:
                    continue

                model.fit(X_train, y_train)
                trained_models[model_name] = model

            if trained_models:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for name, model in trained_models.items():
                        zip_file.writestr(f"{name}_model.pkl", pickle.dumps(model))
                    if scaler:
                        zip_file.writestr("scaler.pkl", pickle.dumps(scaler))
                zip_buffer.seek(0)
                
                st.download_button(label="Download All Models and Scaler",
                                   data=zip_buffer,
                                   file_name="trained_models.zip",
                                   mime="application/zip")

st.set_page_config(page_title="Model Training | Train Models Instantly",page_icon=":robot_face:",layout="centered")

if __name__ == "__main__":
    main()
