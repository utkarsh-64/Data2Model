import streamlit as st
from streamlit_option_menu import option_menu
import home
import preprocess
import model
import eval

st.set_page_config(page_title="Data2Model | One Stop Solution to Train Models",page_icon=":robot_face:",layout="wide",)

# Sidebar navigation
with st.sidebar:
    selected = option_menu("Navigation", 
                           ["Home", "Preprocess Data", "Train Model", "Evaluate Model"], 
                           icons=["house", "tools", "robot", "bar-chart"], 
                           menu_icon="menu-button", 
                           default_index=0)

# Render the selected page
if selected == "Home":
    home.main()
elif selected == "Preprocess Data":
    preprocess.main()
elif selected == "Train Model":
    model.main()
elif selected == "Evaluate Model":
    eval.main()
