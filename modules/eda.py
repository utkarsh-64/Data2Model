import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

def eda_section():
    st.title("Exploratory Data Analysis")
    
    if st.session_state.raw_df is None:
        st.warning("Please upload data first!")
        return
    
    with st.expander("📁 Data Previews", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Raw Data Preview")
            st.dataframe(st.session_state.raw_df.head(), use_container_width=True)
        with col2:
            st.subheader("Processed Data Preview")
            st.dataframe(st.session_state.processed_df.head(), use_container_width=True)
    
    with st.expander("📊 Quick Statistics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Raw Data Statistics")
            st.write(st.session_state.raw_df.describe())
        with col2:
            st.subheader("Processed Data Statistics")
            st.write(st.session_state.processed_df.describe())

    with st.expander("📈 Numerical Analysis", expanded=False):
        numerical_analysis()

    with st.expander("📊 Categorical Analysis", expanded=False):
        categorical_analysis()

    with st.expander("🔗 Correlation Analysis", expanded=False):
        correlation_analysis()

def numerical_analysis():
    df = st.session_state.processed_df
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if not num_cols:
        st.info("No numerical columns found!")
        return
    
    try:
        col1, col2 = st.columns(2)
        with col1:
            selected_num = st.selectbox("Select Numerical Column", num_cols)
            if selected_num:
                st.subheader(f"Distribution of {selected_num}")
                fig = px.histogram(df, x=selected_num, marginal="box", nbins=50)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Scatter Plot")
            col_x = st.selectbox("X Axis", num_cols)
            col_y = st.selectbox("Y Axis", num_cols, index=1 if len(num_cols)>1 else 0)
            if col_x and col_y:
                fig = px.scatter(df, x=col_x, y=col_y)
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Numerical analysis error: {str(e)}")

def categorical_analysis():
    df = st.session_state.processed_df
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not cat_cols:
        st.info("No categorical columns found!")
        return
    
    try:
        col1, col2 = st.columns(2)
        with col1:
            selected_cat = st.selectbox("Select Categorical Column", cat_cols)
            if selected_cat:
                # Fix: Properly name columns after value_counts
                count_df = df[selected_cat].value_counts().reset_index()
                count_df.columns = [selected_cat, 'count']  # Explicit column names
                
                st.subheader(f"Distribution of {selected_cat}")
                fig = px.bar(count_df, 
                            x=selected_cat, 
                            y='count',
                            labels={selected_cat: 'Category', 'count': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Pie Chart")
            pie_col = st.selectbox("Select for Pie Chart", cat_cols)
            if pie_col:
                # Create pie chart with optimized layout
                fig = px.pie(df, 
                            names=pie_col,
                            hole=0.4,
                            labels={'label': 'Category'})
                
                # Adjust layout for better display
                fig.update_layout(
                    legend=dict(
                        title=pie_col,
                        orientation="v",
                        yanchor="middle",
                        y=0.5,
                        xanchor="left",
                        x=1.05,
                        itemwidth=40  # Reduce item width for compact legend
                    ),
                    margin=dict(r=200, t=50, b=50),  # Increase right margin
                    showlegend=True,
                    width=800  # Fixed width for consistent layout
                )
                
                # Add percentage labels outside the chart
                fig.update_traces(
                    pull=[0.02]*len(df[pie_col].unique()),  # Small pull for clarity
                    textposition='outside',
                    texttemplate='%{percent:.1%}',
                    insidetextorientation='radial'
                )
                
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Categorical analysis error: {str(e)}")

def correlation_analysis():
    df = st.session_state.processed_df
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if len(num_cols) < 2:
        st.info("Need at least 2 numerical columns for correlation")
        return
    
    try:
        st.subheader("Correlation Matrix")
        corr_matrix = df[num_cols].corr()
        fig = px.imshow(corr_matrix,
                       x=num_cols,
                       y=num_cols,
                       text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Correlation analysis error: {str(e)}")