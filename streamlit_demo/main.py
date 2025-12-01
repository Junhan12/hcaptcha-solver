"""
Main entry point for Streamlit demo application.
Routes to different page modules based on user selection.
"""
import streamlit as st
import time

# Import page modules
from page_modules import (
    auto_crawl,
    view_eda,
    data_preprocessing,
    data_augmentation,
    model_train,
    create_model,
    model_evaluation,
    demo_upload_inference,
    demo_crawler_solver,
)

# Page configuration
st.set_page_config(page_title="hCAPTCHA Solver", layout="wide")

st.title("hCAPTCHA Solver Workflow")

# Sidebar navigation
st.sidebar.title("Navigation")

# Main sections
main_section = st.sidebar.selectbox(
    "Select Section",
    (
        "1. Auto Crawl Dataset",
        "2. View EDA",
        "3. Data Preprocessing",
        "4. Data Augmentation",
        "5. Model Training",
        "6. Create and Upload Model",
        "7. Model Training Evaluation",
        "8. hCAPTCHA Demo",
    ),
)

# Sub-navigation for hCAPTCHA Demo section
demo_subsection = None
if main_section == "8. hCAPTCHA Demo":
    demo_subsection = st.sidebar.radio(
        "Demo Options",
        (
            "8a. Upload Image for Inference",
            "8b. Auto Crawler, Solver, and Clicker",
        ),
    )

progress = st.progress(0)
status = st.empty()

# Route to appropriate page based on selection
if main_section == "1. Auto Crawl Dataset":
    auto_crawl.render()
elif main_section == "2. View EDA":
    view_eda.render()
elif main_section == "3. Data Preprocessing":
    data_preprocessing.render()
elif main_section == "4. Data Augmentation":
    data_augmentation.render()
elif main_section == "5. Model Training":
    model_train.render()
elif main_section == "6. Create and Upload Model":
    create_model.render()
elif main_section == "7. Model Training Evaluation":
    model_evaluation.render()
elif main_section == "8. hCAPTCHA Demo":
    if demo_subsection == "8a. Upload Image for Inference":
        demo_upload_inference.render(progress, status)
    elif demo_subsection == "8b. Auto Crawler, Solver, and Clicker":
        demo_crawler_solver.render(progress, status)

