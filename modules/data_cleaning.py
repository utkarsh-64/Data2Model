import streamlit as st
import numpy as np
from sklearn.impute import KNNImputer

def data_cleaning_section():
    st.title("Data Cleaning")
    
    if st.session_state.processed_df is None:
        st.warning("Please upload data first!")
        return
    
    df = st.session_state.processed_df
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Missing Values Treatment")
            handle_missing_values(df)
        
        with col2:
            st.subheader("Outlier Management")
            handle_outliers(df)
    
    st.divider()
    
    with st.container():
        st.subheader("Column Operations")
        handle_column_operations(df)
    
    st.session_state.processed_df = df

def handle_missing_values(df):
    null_cols = df.columns[df.isna().any()].tolist()
    
    if not null_cols:
        st.info("No missing values found!")
        return
    
    treatment_method = st.selectbox(
        "Select treatment strategy",
        ["Column-wise Treatment", "Global Treatment", "KNN Imputation"],
        help="Choose between different imputation strategies"
    )

    if treatment_method == "KNN Imputation":
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        selected_cols = st.multiselect("Select numeric columns", numeric_cols)
        n_neighbors = st.number_input("Number of neighbors", 2, 20, 5)
        
        if st.button("Apply KNN Imputation"):
            try:
                imputer = KNNImputer(n_neighbors=n_neighbors)
                df[selected_cols] = imputer.fit_transform(df[selected_cols])
                st.success("KNN imputation applied!")
            except Exception as e:
                st.error(f"KNN imputation failed: {str(e)}")
    
    if treatment_method == "Column-wise Treatment":
        selected_col = st.selectbox("Select column with missing values", null_cols)
        col_method = st.radio(
            f"Treatment for {selected_col}",
            ["Drop NA", "Fill with Mean", "Fill with Median", "Fill with Mode", "Custom Value"],
            horizontal=True
        )
        
        # Get custom value input BEFORE applying treatment
        custom_filler = None
        if col_method == "Custom Value":
            custom_filler = st.text_input("Enter custom value", key=f"custom_{selected_col}")
        
        if st.button("Apply Treatment"):
            if col_method == "Drop NA":
                df.dropna(subset=[selected_col], inplace=True)
            else:
                filler = None
                if col_method == "Fill with Mean":
                    filler = df[selected_col].mean()
                elif col_method == "Fill with Median":
                    filler = df[selected_col].median()
                elif col_method == "Fill with Mode":
                    filler = df[selected_col].mode()[0]
                else:
                    if not custom_filler:
                        st.error("Please enter a custom value first!")
                        return
                    filler = custom_filler
                
                # Preserve original data type
                try:
                    if df[selected_col].dtype != object:
                        filler = type(df[selected_col].iloc[0])(filler)
                except ValueError:
                    st.error("Invalid type for this column. Enter compatible value.")
                    return
                    
                df[selected_col].fillna(filler, inplace=True)
                
            st.success("Missing values treated!")
            st.session_state.processed_df = df.copy()
    
    else:  # Global Treatment
        global_method = st.selectbox(
            "Select global treatment method",
            ["Drop all NA rows", "Fill with column means", "Fill with column medians"]
        )
        
        if st.button("Apply Global Treatment"):
            if global_method == "Drop all NA rows":
                df.dropna(inplace=True)
            elif global_method == "Fill with column means":
                df.fillna(df.mean(), inplace=True)
            else:
                df.fillna(df.median(), inplace=True)
            st.success("Global treatment applied!")

def handle_outliers(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols:
        st.info("No numeric columns for outlier detection!")
        return
    
    selected_cols = st.multiselect(
        "Select columns for outlier treatment",
        numeric_cols,
        help="Choose numeric columns to analyze"
    )
    
    if not selected_cols:
        return
    
    method = st.radio(
        "Detection Method",
        ["Z-Score", "IQR Method", "Percentile-based"],
        horizontal=True
    )
    
    treatment = st.selectbox(
        "Treatment Strategy",
        ["Cap Values", "Remove Outliers", "Log Transformation"]
    )
    
    threshold = st.slider(
        "Detection Threshold",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.5
    )
    
    if st.button("Apply Outlier Treatment"):
        for col in selected_cols:
            data = df[col]
            
            if method == "Z-Score":
                z_scores = np.abs((data - data.mean()) / data.std())
                mask = z_scores > threshold
            elif method == "IQR Method":
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                mask = (data < (Q1 - threshold * IQR)) | (data > (Q3 + threshold * IQR))
            else:
                lower = data.quantile(0.01)
                upper = data.quantile(0.99)
                mask = (data < lower) | (data > upper)
            
            if treatment == "Cap Values":
                lower_bound = data.quantile(0.05)
                upper_bound = data.quantile(0.95)
                df[col] = data.clip(lower_bound, upper_bound)
            elif treatment == "Remove Outliers":
                df = df[~mask]
            else:
                df[col] = np.log1p(data)
        
        st.session_state.processed_df = df
        st.success("Outlier treatment applied!")

def handle_column_operations(df):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Column Removal")
        cols_to_drop = st.multiselect("Select columns to remove", df.columns)
        if cols_to_drop and st.button("Confirm Removal"):
            df.drop(columns=cols_to_drop, inplace=True)
            st.success(f"Removed columns: {', '.join(cols_to_drop)}")
    
    with col2:
        st.subheader("Value Replacement")
        replace_col = st.selectbox("Select column for value replacement", df.columns)
        old_value = st.text_input("Value to replace")
        new_value = st.text_input("Replacement value")
        
        if st.button("Replace Values"):
            try:
                # Get column data type
                col_type = df[replace_col].dtype
                
                # Convert values to match column type
                if np.issubdtype(col_type, np.number):
                    # Handle numerical columns
                    old_val = float(old_value) if '.' in old_value else int(old_value)
                    new_val = float(new_value) if '.' in new_value else int(new_value)
                else:
                    # Handle string columns
                    old_val = str(old_value)
                    new_val = str(new_value)
                
                # Perform replacement
                before_count = (df[replace_col] == old_val).sum()
                df[replace_col] = df[replace_col].replace(old_val, new_val)
                after_count = (df[replace_col] == old_val).sum()
                
                st.success(f"Replaced {before_count - after_count} instances")
                
            except ValueError:
                st.error("Type mismatch! Ensure replacement values match column type")
            except Exception as e:
                st.error(f"Replacement failed: {str(e)}")