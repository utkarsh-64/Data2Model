import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from imblearn.over_sampling import SMOTE

def advanced_preprocessing_section():
    st.title("Advanced Processing")
    
    if st.session_state.processed_df is None:
        st.warning("Please process data in previous steps first!")
        return
    
    df = st.session_state.processed_df
    
    with st.expander("🎯 Target Variable Setup", expanded=True):
        handle_target_variable(df)
    
    with st.expander("📉 Dimensionality Reduction", expanded=False):
        handle_dimensionality_reduction(df)
    
    with st.expander("⚖️ Class Balancing", expanded=False):
        handle_class_balancing(df)
    
    st.session_state.processed_df = df

def handle_target_variable(df):
    st.subheader("Target Column Selection")
    target_col = st.selectbox(
        "Select column to mark as target variable",
        df.columns
    )
    st.session_state.target_col = target_col
    st.info(f"Selected target column: {target_col}")

def handle_dimensionality_reduction(df):
    if not st.session_state.target_col:
        st.warning("Select target variable first!")
        return
    
    method = st.selectbox(
        "Reduction Method",
        ["PCA", "Feature Importance", "Correlation Threshold"]
    )
    
    if method == "PCA":
        n_components = st.slider(
            "Number of Components",
            1, df.shape[1]-1, 3
        )
        if st.button("Apply PCA"):
            pca = PCA(n_components=n_components)
            features = df.drop(st.session_state.target_col, axis=1)
            pca_features = pca.fit_transform(features)
            pca_df = pd.DataFrame(
                pca_features,
                columns=[f"PC_{i+1}" for i in range(n_components)]
            )
            final_df = pd.concat(
                [pca_df, df[[st.session_state.target_col]]],
                axis=1
            )
            st.session_state.processed_df = final_df
            st.success(f"Explained Variance: {np.sum(pca.explained_variance_ratio_):.2%}")
    
    elif method == "Feature Importance":
        k = st.slider("Number of Features", 1, df.shape[1]-1, 10)
        score_func = st.selectbox(
            "Scoring Function",
            ["ANOVA F-value", "Chi-Square"]
        )
        
        if st.button("Select Features"):
            X = df.drop(st.session_state.target_col, axis=1)
            y = df[st.session_state.target_col]
            
            if score_func == "Chi-Square":
                selector = SelectKBest(chi2, k=k)
            else:
                selector = SelectKBest(f_classif, k=k)
            
            selector.fit(X, y)
            selected_cols = X.columns[selector.get_support()]
            final_df = df[selected_cols.tolist() + [st.session_state.target_col]]
            st.session_state.processed_df = final_df
            st.success(f"Selected {len(selected_cols)} features")

def handle_class_balancing(df):
    if st.session_state.problem_type != "Classification":
        st.info("Class balancing only for classification tasks")
        return
    
    target = st.session_state.target_col
    class_counts = df[target].value_counts()
    
    st.write("Current Class Distribution:")
    st.bar_chart(class_counts)
    
    balance_method = st.selectbox(
        "Balancing Method",
        ["SMOTE (Oversampling)", "Undersampling", "Class Weights"]
    )
    
    if st.button("Balance Classes"):
        X = df.drop(target, axis=1)
        y = df[target]
        
        if balance_method == "SMOTE (Oversampling)":
            smote = SMOTE()
            X_res, y_res = smote.fit_resample(X, y)
        elif balance_method == "Undersampling":
            min_class = class_counts.idxmin()
            min_count = class_counts.min()
            resampled = [df[df[target] == min_class]]
            for cls, count in class_counts.items():
                if cls != min_class:
                    resampled.append(df[df[target] == cls].sample(min_count))
            resampled_df = pd.concat(resampled)
            X_res = resampled_df.drop(target, axis=1)
            y_res = resampled_df[target]
        
        balanced_df = pd.concat([X_res, y_res], axis=1)
        st.session_state.processed_df = balanced_df
        st.success(f"New distribution: {y_res.value_counts().to_dict()}")