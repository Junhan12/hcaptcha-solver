"""
Data Preprocessing page.
"""
import streamlit as st
import io
import traceback
import pandas as pd
from PIL import Image

import sys
import os
_this_dir = os.path.dirname(__file__)
_parent_dir = os.path.abspath(os.path.join(_this_dir, '..'))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from utils import (
    list_models,
    get_model_by_id,
    get_preprocess_for_model,
    apply_preprocess,
)


def render():
    """Render the Data Preprocessing page."""
    st.header("Data Preprocessing")
    st.info("Select a model and upload images to apply the model's preprocessing steps.")
    
    # Model selection
    try:
        models = list_models(limit=100)
        if not models:
            st.warning("No models found in MongoDB. Please create a model first in the 'Create and Upload Model' section.")
        else:
            model_options = {f"{m.get('model_name', 'Unknown')} ({m.get('model_id', 'N/A')})": m.get('model_id') for m in models}
            selected_preprocess_model_name = st.selectbox(
                "Select Model",
                options=list(model_options.keys()),
                key="model_preprocess_select"
            )
            selected_preprocess_model_id = model_options[selected_preprocess_model_name]
            selected_preprocess_model = get_model_by_id(selected_preprocess_model_id)
            
            if selected_preprocess_model:
                st.markdown("### Model Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"**Model ID:** {selected_preprocess_model.get('model_id', 'N/A')}")
                with col2:
                    st.caption(f"**Model Name:** {selected_preprocess_model.get('model_name', 'N/A')}")
                
                # Get preprocessing profile for the selected model
                preprocess_profile = get_preprocess_for_model(selected_preprocess_model)
                
                if preprocess_profile:
                    st.markdown("### Preprocessing Profile")
                    st.info(f"**Preprocessing ID:** {preprocess_profile.get('preprocess_id', 'N/A')}")
                    if preprocess_profile.get('name'):
                        st.caption(f"**Name:** {preprocess_profile.get('name', 'N/A')}")
                    
                    steps = preprocess_profile.get('steps', [])
                    if steps:
                        st.caption(f"**Number of Steps:** {len(steps)}")
                        with st.expander("View Preprocessing Steps"):
                            for idx, step in enumerate(steps, 1):
                                operation = step.get('operation', 'Unknown')
                                params = step.get('params', {})
                                st.write(f"{idx}. **{operation}**")
                                if params:
                                    st.json(params)
                    else:
                        st.warning("⚠️ This model has a preprocessing profile but no steps defined.")
                else:
                    st.warning("⚠️ This model does not have a preprocessing profile configured.")
                
                st.markdown("---")
                
                # Image upload section
                st.markdown("### Upload Image for Preprocessing")
                uploaded_image = st.file_uploader(
                    "Upload Image",
                    type=["jpg", "jpeg", "png", "bmp"],
                    key="preprocess_image_upload"
                )
                
                if uploaded_image and preprocess_profile:
                    if st.button("Apply Preprocessing", key="apply_preprocess_button"):
                        with st.spinner("Applying preprocessing steps..."):
                            try:
                                # Read original image bytes
                                original_img_bytes = uploaded_image.read()
                                
                                # Apply preprocessing
                                processed_img_bytes, applied_steps = apply_preprocess(original_img_bytes, preprocess_profile)
                                
                                # Display results
                                st.markdown("### Preprocessing Results")
                                
                                # Load images for display
                                original_img = Image.open(io.BytesIO(original_img_bytes))
                                processed_img = Image.open(io.BytesIO(processed_img_bytes))
                                
                                # Get processed image dimensions (for sizing)
                                processed_width, processed_height = processed_img.size
                                
                                # Display images side by side
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("#### Original Image")
                                    st.image(
                                        original_img,
                                        caption=f"Original ({original_img.size[0]} × {original_img.size[1]})",
                                        width=processed_width  # Use processed image width
                                    )
                                
                                with col2:
                                    st.markdown("#### Preprocessed Image")
                                    st.image(
                                        processed_img,
                                        caption=f"Preprocessed ({processed_width} × {processed_height})",
                                        width=processed_width
                                    )
                                
                                # Display preprocessing steps applied
                                st.markdown("### Applied Preprocessing Steps")
                                if applied_steps:
                                    steps_data = []
                                    for idx, step in enumerate(applied_steps, 1):
                                        operation = step.get('operation', 'Unknown')
                                        params = step.get('params', {})
                                        steps_data.append({
                                            "Step": idx,
                                            "Operation": operation,
                                            "Parameters": str(params) if params else "None"
                                        })
                                    
                                    df_steps = pd.DataFrame(steps_data)
                                    st.dataframe(df_steps, width='stretch', hide_index=True)
                                else:
                                    st.info("No preprocessing steps were applied (or preprocessing profile has no steps).")
                                
                                # Image size comparison
                                st.markdown("### Image Size Comparison")
                                size_col1, size_col2 = st.columns(2)
                                with size_col1:
                                    st.metric("Original Size", f"{original_img.size[0]} × {original_img.size[1]}")
                                with size_col2:
                                    st.metric("Processed Size", f"{processed_width} × {processed_height}")
                                
                            except Exception as e:
                                st.error(f"Error applying preprocessing: {e}")
                                st.code(traceback.format_exc())
                
                elif uploaded_image and not preprocess_profile:
                    st.error("⚠️ Cannot apply preprocessing: Selected model does not have a preprocessing profile.")
                
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.code(traceback.format_exc())

