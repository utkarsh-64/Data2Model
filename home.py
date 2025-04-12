import streamlit as st

def main():
    st.title("Welcome to Data 2 Model 🧠")
    st.write("### Your One-Stop Solution for Machine Learning Model Training!")

    col1, col2 = st.columns(2)

    with col1:
        st.image("./Image.jpg", caption="Data to Model in Action")
        st.image("./Image2.png", caption="Machine Learning Process")

    with col2:
        st.markdown(
            """
            **Why Choose Data 2 Model?**
            - 🚀 **Preprocess Your Data**: Clean, scale, and transform datasets.
            - 🤖 **Train Models Easily**: Choose from multiple ML models.
            - 📊 **Evaluate Performance**: Get instant insights and visualizations.
            - 💾 **Download Models**: Save trained models for deployment.
            """
        )

    st.write("### Get Started in 3 Simple Steps")
    st.write("1️⃣ **Go to 'Preprocess Data'** and upload your dataset.")
    st.write("2️⃣ **Train your model** using our automated selection.")
    st.write("3️⃣ **Evaluate performance** and improve results.")

    st.success("Ready? Click on the Sidebar to Begin! 🚀")
