import streamlit as st
import pandas as pd


def data_upload_section():
    st.title("Data Ingestion")
    
    col1, col2 = st.columns([3, 2])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV File", 
            type=["csv"],
            help="Maximum file size: 200MB"
        )
        
        if uploaded_file:
            st.session_state.raw_df = None
            if st.session_state.raw_df is None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.raw_df = df.copy()
                    st.session_state.processed_df = df.copy()
                    st.success("Data loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
    
    with col2:
        if st.session_state.raw_df is not None:
            st.subheader("Dataset Summary")
            st.write(f"Rows: {st.session_state.raw_df.shape[0]}")
            st.write(f"Columns: {st.session_state.raw_df.shape[1]}")
            st.write(f"Missing Values: {st.session_state.raw_df.isna().sum().sum()}")
            st.write(f"Duplicate Rows: {st.session_state.raw_df.duplicated().sum()}")
    
    if st.session_state.raw_df is not None:
        with st.expander("Preview Raw Data"):
            st.dataframe(st.session_state.raw_df.head(10), use_container_width=True)
        
        with st.expander("Data Types Overview"):
            dtype_df = pd.DataFrame(st.session_state.raw_df.dtypes, columns=["Data Type"])
            st.dataframe(dtype_df, use_container_width=True)
        
        with st.expander("Column Statistics"):
            col_to_analyze = st.selectbox(
                "Select column for detailed analysis",
                st.session_state.raw_df.columns
            )
            if col_to_analyze:
                col_data = st.session_state.raw_df[col_to_analyze]
                st.write("**Basic Statistics**")
                st.write(col_data.describe())
                
                if pd.api.types.is_numeric_dtype(col_data):
                    st.write("**Distribution**")
                    st.bar_chart(col_data)
                else:
                    st.write("**Value Counts**")
                    st.write(col_data.value_counts())