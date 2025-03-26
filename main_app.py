import streamlit as st
from modules.data_upload import data_upload_section
from modules.data_cleaning import data_cleaning_section
from modules.feature_engineering import feature_engineering_section
from modules.advanced_preprocess import advanced_preprocessing_section
from modules.utils import export_section
from modules.eda import eda_section

def main():
    st.set_page_config(
        page_title="DataPrep Studio",
        page_icon="🧊",
        layout="wide"
    )
    
    # Initialize session state
    if 'raw_df' not in st.session_state:
        st.session_state.raw_df = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
        
    # Sidebar Navigation
    with st.sidebar:
        st.title("DataPrep Studio")
        nav_options = [
            "Data Upload",
            "Exploratory Data Analysis",
            "Data Cleaning",
            "Feature Engineering",
            "Export Data"
        ]
        selected_nav = st.selectbox(
            "Workflow Steps", 
            nav_options,
            index=0  # Default to first option
        )
        
        if st.session_state.processed_df is not None:
            st.divider()
            st.markdown(f"**Data Shape:** {st.session_state.processed_df.shape}")
            if 'target_col' in st.session_state:
                st.markdown(f"**Target Column:** {st.session_state.target_col}")
    
    # Main content routing
    if selected_nav == "Data Upload":
        data_upload_section()
    elif selected_nav == "Exploratory Data Analysis":
        eda_section()
    elif selected_nav == "Data Cleaning":
        data_cleaning_section()
    elif selected_nav == "Feature Engineering":
        feature_engineering_section()
    elif selected_nav == "Export Data":
        export_section()

if __name__ == "__main__":
    main()