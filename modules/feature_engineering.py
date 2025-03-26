import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import PolynomialFeatures

def feature_engineering_section():
    st.title("Feature Engineering")
    
    if st.session_state.processed_df is None:
        st.warning("Please upload and clean data first!")
        return
    
    df = st.session_state.processed_df
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("🔄 Categorical Encoding", expanded=True):
                handle_categorical_encoding(df)
            
            with st.expander("📅 DateTime Features", expanded=True):
                handle_datetime_features(df)
        
        with col2:
            with st.expander("🧮 Feature Creation", expanded=True):
                handle_feature_creation(df)
            
            with st.expander("📊 Binning & Discretization", expanded=True):
                handle_binning(df)
    
    with st.expander("⚛️ Polynomial Features", expanded=False):
        handle_polynomial_features(df)
    
    st.session_state.processed_df = df

def handle_categorical_encoding(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not cat_cols:
        st.info("No categorical columns found!")
        return
    
    col1, col2 = st.columns([3, 2])
    with col1:
        selected_col = st.selectbox("Select categorical column", cat_cols)
    with col2:
        encode_method = st.selectbox(  # Changed to selectbox
            "Encoding Method",
            ["One-Hot", "Label", "Frequency"]
        )
    
    if st.button("Apply Encoding"):
        try:
            if encode_method == "One-Hot":
                # Remove existing dummies first
                prefix = f"{selected_col}_"
                existing_dummies = [col for col in df.columns if col.startswith(prefix)]
                df = df.drop(columns=existing_dummies)
                
                # Create new dummies
                dummies = pd.get_dummies(df[selected_col], prefix=prefix)
                df = pd.concat([df, dummies], axis=1)
                st.session_state.encode_message = f"✅ Successfully created {dummies.shape[1]} one-hot encoded columns!"
            
            elif encode_method == "Label":
                # Create new encoded column
                le = LabelEncoder()
                df[f"{selected_col}_encoded"] = le.fit_transform(df[selected_col])
                st.success("Label encoding applied!")
                st.write("Encoded Values:")
                st.write(df[f"{selected_col}_encoded"].value_counts())
            
            else:  # Frequency encoding
                # Create new frequency column
                freq = df[selected_col].value_counts(normalize=True)
                df[f"{selected_col}_freq"] = df[selected_col].map(freq)
                st.success("Frequency encoding applied!")
                st.write("Encoded Values Distribution:")
                st.write(df[f"{selected_col}_freq"].describe())
            
            # Update session state without rerun
            st.session_state.processed_df = df.copy()
            
        except Exception as e:
            st.error(f"Encoding failed: {str(e)}")

        st.session_state.processed_df = df.copy()
        st.rerun()     

def handle_datetime_features(df):
    datetime_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    
    if not datetime_cols:
        if st.checkbox("Convert column to datetime"):
            selected_col = st.selectbox("Select text date column", df.columns)
            try:
                df[selected_col] = pd.to_datetime(df[selected_col])
                datetime_cols.append(selected_col)
                st.success("Column converted to datetime!")
            except:
                st.error("Could not convert to datetime")
        return
    
    selected_col = st.selectbox("Select datetime column", datetime_cols)
    
    date_options = st.multiselect(
        "Extract features",
        ["Year", "Month", "Day", "Weekday", "Hour", "Minute", "Quarter"],
        help="Select temporal components to extract"
    )
    
    if date_options:
        if st.button("Extract Date Features"):
            for option in date_options:
                if option == "Year":
                    df[f"{selected_col}_year"] = df[selected_col].dt.year
                elif option == "Month":
                    df[f"{selected_col}_month"] = df[selected_col].dt.month
                elif option == "Day":
                    df[f"{selected_col}_day"] = df[selected_col].dt.day
                elif option == "Weekday":
                    df[f"{selected_col}_weekday"] = df[selected_col].dt.weekday
                elif option == "Hour":
                    df[f"{selected_col}_hour"] = df[selected_col].dt.hour
                elif option == "Minute":
                    df[f"{selected_col}_minute"] = df[selected_col].dt.minute
                elif option == "Quarter":
                    df[f"{selected_col}_quarter"] = df[selected_col].dt.quarter
            
            st.success(f"Added {len(date_options)} date features!")

def handle_feature_creation(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    col1, col2, col3 = st.columns([3, 3, 2])
    with col1:
        col_a = st.selectbox("First column", numeric_cols)
    with col2:
        operation = st.selectbox("Operation", ["+", "-", "*", "/", "log"])
    with col3:
        col_b = st.selectbox("Second column", [""] + numeric_cols)
    
    new_feat_name = st.text_input("New feature name")
    
    if operation == "log":
        if st.button("Create Logarithmic Feature"):
            try:
                df[new_feat_name] = np.log1p(df[col_a])
                st.success("Log feature created!")
            except:
                st.error("Invalid log operation")
    elif col_b:
        if st.button("Create Mathematical Feature"):
            try:
                if operation == "+":
                    df[new_feat_name] = df[col_a] + df[col_b]
                elif operation == "-":
                    df[new_feat_name] = df[col_a] - df[col_b]
                elif operation == "*":
                    df[new_feat_name] = df[col_a] * df[col_b]
                elif operation == "/":
                    df[new_feat_name] = df[col_a] / df[col_b]
                st.success("Feature created successfully!")
            except:
                st.error("Could not perform operation")

def handle_binning(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols:
        st.info("No numeric columns for binning!")
        return
    
    selected_col = st.selectbox("Select numeric column", numeric_cols)
    bin_method = st.radio(
        "Binning Method",
        ["Equal Width", "Equal Frequency", "Custom Ranges"],
        horizontal=True
    )
    
    if bin_method == "Custom Ranges":
        bins = st.text_input("Enter bin edges (comma-separated)")
        try:
            bins = [float(x.strip()) for x in bins.split(",")]
        except:
            st.error("Invalid bin format")
    else:
        n_bins = st.number_input("Number of bins", 2, 20, 5)
    
    new_col_name = st.text_input("Binned column name")
    
    if st.button("Apply Binning"):
        try:
            if bin_method == "Equal Width":
                df[new_col_name] = pd.cut(df[selected_col], bins=n_bins)
            elif bin_method == "Equal Frequency":
                df[new_col_name] = pd.qcut(df[selected_col], q=n_bins)
            else:
                df[new_col_name] = pd.cut(df[selected_col], bins=bins)
            
            st.success(f"{bin_method} binning applied!")
        except Exception as e:
            st.error(f"Binning failed: {str(e)}")

def handle_polynomial_features(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    selected_cols = st.multiselect(
        "Select numeric columns for polynomial features",
        numeric_cols,
        help="Choose 2-5 columns for best results"
    )
    
    if len(selected_cols) < 2:
        return
    
    degree = st.slider("Polynomial Degree", 2, 4, 2)
    interaction_only = st.checkbox("Interaction terms only")
    
    if st.button("Generate Polynomial Features"):
        try:
            poly = PolynomialFeatures(
                degree=degree,
                interaction_only=interaction_only,
                include_bias=False
            )
            poly_features = poly.fit_transform(df[selected_cols])
            feature_names = poly.get_feature_names_out(selected_cols)
            poly_df = pd.DataFrame(poly_features, columns=feature_names)
            df = pd.concat([df, poly_df], axis=1)
            st.success(f"Added {poly_df.shape[1]} polynomial features!")
        except Exception as e:
            st.error(f"Polynomial features failed: {str(e)}")