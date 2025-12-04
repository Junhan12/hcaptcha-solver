"""
Create and Upload Model page.
"""
import streamlit as st
import requests
import traceback

import sys
import os
_this_dir = os.path.dirname(__file__)
_parent_dir = os.path.abspath(os.path.join(_this_dir, '..'))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from utils import (
    API_TIMEOUT,
    list_models,
    get_model_by_id,
    delete_model,
    list_preprocess_profiles,
    list_postprocess_profiles,
    upsert_model,
)
import pandas as pd
import re


def get_next_model_id():
    """
    Get the next model ID by finding the highest model_id in the database
    and incrementing it. Format: m-001, m-002, etc.
    """
    try:
        models = list_models(limit=1000)  # Get all models to find max
        if not models:
            return "m-001"
        
        # Extract numeric parts from model_ids
        max_num = 0
        for model in models:
            model_id = model.get('model_id', '')
            # Match pattern like "m-001", "m-123", etc.
            match = re.match(r'm-(\d+)', model_id, re.IGNORECASE)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)
        
        # Increment and format
        next_num = max_num + 1
        return f"m-{next_num:03d}"
    except Exception as e:
        # Fallback to m-001 if error
        return "m-001"


def render():
    """Render the Create and Upload Model page."""
    # Use existing "Create/Update Model" functionality
    st.subheader("Create / Update Model")
    st.info("Evaluation results will be automatically updated when you run the Model Training Evaluation.")
    
    # Load profiles outside form for use in submission
    try:
        preprocess_profiles = list_preprocess_profiles(limit=100)
        postprocess_profiles = list_postprocess_profiles(limit=100)
    except Exception as e:
        st.warning(f"Could not load profiles: {e}")
        preprocess_profiles = []
        postprocess_profiles = []
    
    # Auto-generate next model ID
    auto_model_id = get_next_model_id()
    
    with st.form("model_form"):
        model_id = st.text_input("Model ID", value=auto_model_id, disabled=True, help="Auto-generated model ID")
        model_name = st.text_input("Model Name", placeholder="yolov8-object-001")
        weights = st.file_uploader("Weights (.pt)", type=["pt"])
        is_active = st.checkbox("Set Active", value=False)
        
        # Preprocessing profile selection
        st.markdown("#### Preprocessing Profile (Optional)")
        preprocess_options = ["None"] + [f"{p.get('name', 'Unnamed')} ({p.get('preprocess_id', 'N/A')})" for p in preprocess_profiles]
        selected_preprocess_display = st.selectbox(
            "Select Preprocessing Profile",
            options=preprocess_options,
            key="preprocess_select"
        )
        
        # Postprocessing profile selection
        st.markdown("#### Postprocessing Profile (Optional)")
        postprocess_options = ["None"] + [f"{p.get('name', 'Unnamed')} ({p.get('postprocess_id', 'N/A')})" for p in postprocess_profiles]
        selected_postprocess_display = st.selectbox(
            "Select Postprocessing Profile",
            options=postprocess_options,
            key="postprocess_select"
        )
        
        submitted = st.form_submit_button("Save Model")

    if submitted:
        # Use the auto-generated model_id from the form (even though disabled, value is still available)
        if not model_name:
            st.error("model_name is required")
        else:
            # model_id is already set from the form input above
            # Extract preprocess_id and postprocess_id from selected display strings
            # Access form values from session_state
            selected_preprocess_display = st.session_state.get("preprocess_select", "None")
            selected_postprocess_display = st.session_state.get("postprocess_select", "None")
            
            selected_preprocess_id = None
            if selected_preprocess_display != "None":
                for p in preprocess_profiles:
                    display_str = f"{p.get('name', 'Unnamed')} ({p.get('preprocess_id', 'N/A')})"
                    if display_str == selected_preprocess_display:
                        selected_preprocess_id = p.get('preprocess_id')
                        break
            
            selected_postprocess_id = None
            if selected_postprocess_display != "None":
                for p in postprocess_profiles:
                    display_str = f"{p.get('name', 'Unnamed')} ({p.get('postprocess_id', 'N/A')})"
                    if display_str == selected_postprocess_display:
                        selected_postprocess_id = p.get('postprocess_id')
                        break
            
            data = {
                "model_id": model_id,
                "model_name": model_name,
                "is_active": "true" if is_active else "false",
            }
            
            files = {}
            if weights is not None:
                files = {"weights": (weights.name, weights, "application/octet-stream")}
            
            try:
                # If weights are provided, use API endpoint
                if weights is not None:
                    resp = requests.post("http://localhost:5000/models", data=data, files=files, timeout=API_TIMEOUT)
                    if resp.ok:
                        # Update preprocess_id and postprocess_id after model is created
                        # Note: If weights are uploaded, the model creation happens in background
                        # So we wait a moment and then update the profile IDs
                        if selected_preprocess_id or selected_postprocess_id:
                            import time
                            time.sleep(0.5)  # Small delay to ensure model document exists
                            from app.database import _db, _db_available
                            if _db_available():
                                update_doc = {}
                                if selected_preprocess_id:
                                    update_doc["preprocess_id"] = selected_preprocess_id
                                if selected_postprocess_id:
                                    update_doc["postprocess_id"] = selected_postprocess_id
                                if update_doc:
                                    # Retry a few times in case model document isn't ready yet
                                    max_retries = 3
                                    for attempt in range(max_retries):
                                        result = _db.model.update_one({"model_id": model_id}, {"$set": update_doc})
                                        if result.matched_count > 0:
                                            break
                                        if attempt < max_retries - 1:
                                            time.sleep(0.5)
                        st.success("Model saved.")
                    else:
                        st.error(f"Failed: {resp.status_code} {resp.text}")
                else:
                    # If no weights, use direct database call to update metadata including profiles
                    weights_stream = None
                    results = {}
                    updated_model = upsert_model(
                        model_id=model_id,
                        model_name=model_name,
                        weights_file_stream=weights_stream,
                        results=results,
                        is_active=is_active
                    )
                    if updated_model:
                        # Update preprocess_id and postprocess_id separately if needed
                        from app.database import _db, _db_available
                        if _db_available():
                            update_doc = {}
                            if selected_preprocess_id:
                                update_doc["preprocess_id"] = selected_preprocess_id
                            if selected_postprocess_id:
                                update_doc["postprocess_id"] = selected_postprocess_id
                            if update_doc:
                                _db.model.update_one({"model_id": model_id}, {"$set": update_doc})
                        st.success("Model saved.")
                    else:
                        st.error("Failed to save model.")
            except Exception as e:
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())

    st.markdown("---")
    
    # Delete Model Section
    st.subheader("Delete Model")
    st.info("Select a model from MongoDB to delete. This will remove the model document and its associated weights from GridFS.")
    
    try:
        models = list_models(limit=100)
        if not models:
            st.warning("No models found in MongoDB.")
        else:
            model_options = {f"{m.get('model_name', 'Unknown')} ({m.get('model_id', 'N/A')})": m.get('model_id') for m in models}
            selected_delete_model_name = st.selectbox(
                "Select Model to Delete",
                options=list(model_options.keys()),
                key="model_delete_select"
            )
            selected_delete_model_id = model_options[selected_delete_model_name]
            selected_delete_model = get_model_by_id(selected_delete_model_id)
            
            if selected_delete_model:
                st.markdown("#### Model Details")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"**Model ID:** {selected_delete_model.get('model_id', 'N/A')}")
                with col2:
                    st.caption(f"**Model Name:** {selected_delete_model.get('model_name', 'N/A')}")
                with col3:
                    st.caption(f"**Is Active:** {'Yes' if selected_delete_model.get('is_active') else 'No'}")
                
                if selected_delete_model.get('weights'):
                    st.caption(f"This model has weights stored in GridFS that will also be deleted.")
                
                # Confirmation before deletion
                confirm_delete = st.checkbox(
                    f"I confirm I want to delete model '{selected_delete_model.get('model_id', 'N/A')}'",
                    key="confirm_delete_checkbox"
                )
                
                if st.button("Delete Model", type="primary", disabled=not confirm_delete, key="delete_model_button"):
                    with st.spinner("Deleting model..."):
                        result = delete_model(selected_delete_model_id)
                        if result['success']:
                            st.success(result['message'])
                            st.rerun()  # Refresh the page to update the model list
                        else:
                            st.error(result['message'])
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.code(traceback.format_exc())
    
    st.markdown("---")
    
    # List Models Section
    st.subheader("List Models")
    if st.button("List Models", type="primary"):
        try:
            models = list_models(limit=100)
            if not models:
                st.warning("No models found in MongoDB.")
            else:
                # Prepare data for table
                models_data = []
                for model in models:
                    models_data.append({
                        "Model ID": model.get('model_id', 'N/A'),
                        "Model Name": model.get('model_name', 'N/A')
                    })
                
                # Display as table
                df_models = pd.DataFrame(models_data)
                st.dataframe(
                    df_models,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Model ID": st.column_config.TextColumn(
                            "Model ID",
                            help="Unique model identifier",
                            width="medium"
                        ),
                        "Model Name": st.column_config.TextColumn(
                            "Model Name",
                            help="Name of the model",
                            width="large"
                        )
                    }
                )
        except Exception as e:
            st.error(f"Error: {e}")
            st.code(traceback.format_exc())

