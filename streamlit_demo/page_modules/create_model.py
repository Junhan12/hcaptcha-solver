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
    clear_model_cache,
    get_cache_info,
    list_preprocess_profiles,
    list_postprocess_profiles,
    upsert_model,
)


def render():
    """Render the Create and Upload Model page."""
    # Use existing "Create/Update Model" functionality
    st.subheader("Create / Update Model")
    st.info("💡 Evaluation results will be automatically updated when you run the Model Training Evaluation.")
    
    # Load profiles outside form for use in submission
    try:
        preprocess_profiles = list_preprocess_profiles(limit=100)
        postprocess_profiles = list_postprocess_profiles(limit=100)
    except Exception as e:
        st.warning(f"Could not load profiles: {e}")
        preprocess_profiles = []
        postprocess_profiles = []
    
    with st.form("model_form"):
        model_id = st.text_input("Model ID", placeholder="m-001")
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
        if not model_id or not model_name:
            st.error("model_id and model_name are required")
        else:
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
    
    # Model cache management section
    st.subheader("Model Cache Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("View Cache Info"):
            try:
                cache_info = get_cache_info()
                st.json({
                    "Cached Models": cache_info['cached_models'],
                    "Model Count": cache_info['model_count'],
                    "Weights Count": cache_info['weights_count'],
                    "Temp Files": cache_info['temp_file_count']
                })
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        if st.button("Clear All Model Cache"):
            try:
                cleared = clear_model_cache()
                st.success(f"Cache cleared! Removed {cleared['models']} models, {cleared['weights']} weights, and {cleared['temp_files']} temp files.")
                if cleared['temp_file_paths']:
                    st.caption(f"Deleted temp files: {len(cleared['temp_file_paths'])} files")
            except Exception as e:
                st.error(f"Error: {e}")
    
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
                    st.caption(f"⚠️ This model has weights stored in GridFS that will also be deleted.")
                
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
    if st.button("List Models"):
        try:
            resp = requests.get("http://localhost:5000/models", timeout=API_TIMEOUT)
            if resp.ok:
                items = resp.json().get("items", [])
                for it in items:
                    st.write({
                        "model_id": it.get("model_id"),
                        "model_name": it.get("model_name"),
                        "is_active": it.get("is_active"),
                        "results": it.get("results"),
                        "weights": it.get("weights"),
                    })
            else:
                st.error(f"List failed: {resp.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")

