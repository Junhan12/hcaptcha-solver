import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import base64
import time
import io
from datetime import datetime

# Ensure project root is on sys.path to import client and app modules
_this_dir = os.path.dirname(__file__)
_project_root = os.path.abspath(os.path.join(_this_dir, '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from client.crawler import run_crawl_once
from app import decompress_image_to_base64
from app.config import API_TIMEOUT
from app.database import list_models, get_model_by_id, get_preprocess_for_model, get_postprocess_for_model, upsert_model

# Import delete_model with error handling for Streamlit caching issues
try:
    from app.database import delete_model
except ImportError:
    # If import fails, try to reload the module
    import importlib
    import app.database
    importlib.reload(app.database)
    from app.database import delete_model
from app.solver import solve_captcha

# Import cache functions - handle potential cache issues
try:
    from app.solver import clear_model_cache, get_cache_info
except ImportError:
    # If import fails, try to reload the module
    import importlib
    import app.solver
    importlib.reload(app.solver)
    from app.solver import clear_model_cache, get_cache_info

from app.preprocess import apply_preprocess

# Import evaluator functions - handle potential cache issues
try:
    from app.evaluator import evaluate_model, load_validation_dataset, parse_data_yaml
except ImportError as e:
    # If import fails, try to reload the module
    import importlib
    import app.evaluator
    importlib.reload(app.evaluator)
    from app.evaluator import evaluate_model, load_validation_dataset, parse_data_yaml

import json
import pandas as pd
import yaml

# Try to import plotly for visualization
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

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
        "5. Create and Upload Model",
        "6. Model Training Evaluation",
        "7. hCAPTCHA Demo",
    ),
)

# Sub-navigation for hCAPTCHA Demo section
demo_subsection = None
if main_section == "7. hCAPTCHA Demo":
    demo_subsection = st.sidebar.radio(
        "Demo Options",
        (
            "7a. Upload Image for Inference",
            "7b. Auto Crawler, Solver, and Clicker",
    ),
)

progress = st.progress(0)
status = st.empty()

# Route to appropriate page based on selection
if main_section == "1. Auto Crawl Dataset":
    st.header("Auto Crawl Dataset")
    st.info("This section will allow you to automatically crawl hCAPTCHA challenges and collect dataset images.")
    st.write("🚧 **Feature coming soon** - This will enable automated dataset collection from hCAPTCHA challenges.")
    
elif main_section == "2. View EDA":
    st.header("View EDA (Exploratory Data Analysis)")
    st.info("This section provides exploratory data analysis of your collected dataset.")
    st.write("🚧 **Feature coming soon** - This will display statistics, visualizations, and insights about your dataset.")
    
elif main_section == "3. Data Preprocessing":
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
                                    
                                    import pandas as pd
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
                                import traceback
                                st.code(traceback.format_exc())
                
                elif uploaded_image and not preprocess_profile:
                    st.error("⚠️ Cannot apply preprocessing: Selected model does not have a preprocessing profile.")
                
    except Exception as e:
        st.error(f"Error loading models: {e}")
        import traceback
        st.code(traceback.format_exc())
    
elif main_section == "4. Data Augmentation":
    st.header("Data Augmentation")
    st.info("Apply data augmentation techniques to expand your training dataset.")
    st.write("🚧 **Feature coming soon** - This will provide data augmentation options for training data expansion.")
    
elif main_section == "5. Create and Upload Model":
    # Use existing "Create/Update Model" functionality
    st.subheader("Create / Update Model")
    st.info("💡 Evaluation results will be automatically updated when you run the Model Training Evaluation.")
    
    with st.form("model_form"):
        model_id = st.text_input("Model ID", placeholder="m-001")
        model_name = st.text_input("Model Name", placeholder="yolov8-object-001")
        weights = st.file_uploader("Weights (.pt)", type=["pt"])
        is_active = st.checkbox("Set Active", value=False)
        submitted = st.form_submit_button("Save Model")

    if submitted:
        if not model_id or not model_name:
            st.error("model_id and model_name are required")
        else:
            data = {
                "model_id": model_id,
                "model_name": model_name,
                "is_active": "true" if is_active else "false",
            }
            files = {}
            if weights is not None:
                files = {"weights": (weights.name, weights, "application/octet-stream")}
            try:
                resp = requests.post("http://localhost:5000/models", data=data, files=files, timeout=API_TIMEOUT)
                if resp.ok:
                    st.success("Model saved.")
                else:
                    st.error(f"Failed: {resp.status_code} {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")

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
        import traceback
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

elif main_section == "6. Model Training Evaluation":
    st.header("Model Training Evaluation")
    st.info("Select a model from MongoDB and evaluate it using ground truth annotations.")
    
    # Model selection
    try:
        models = list_models(limit=100)
        if not models:
            st.warning("No models found in MongoDB. Please create a model first in the 'Create and Upload Model' section.")
        else:
            model_options = {f"{m.get('model_name', 'Unknown')} ({m.get('model_id', 'N/A')})": m.get('model_id') for m in models}
            selected_model_name = st.selectbox(
                "Select Model",
                options=list(model_options.keys()),
                key="model_eval_select"
            )
            selected_model_id = model_options[selected_model_name]
            selected_model = get_model_by_id(selected_model_id)
            
            if selected_model:
                st.markdown("### Model Information")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model ID", selected_model.get('model_id', 'N/A'))
                with col2:
                    st.metric("Model Name", selected_model.get('model_name', 'N/A'))
                with col3:
                    st.metric("Active", "Yes" if selected_model.get('is_active', False) else "No")
                
                # Display existing metrics if available
                existing_results = selected_model.get('results', {})
                if existing_results:
                    st.markdown("#### Existing Metrics (from database)")
                    res_col1, res_col2, res_col3, res_col4, res_col5 = st.columns(5)
                    with res_col1:
                        st.metric("Precision", f"{existing_results.get('precision', 0):.4f}" if existing_results.get('precision') else "N/A")
                    with res_col2:
                        st.metric("Recall", f"{existing_results.get('recall', 0):.4f}" if existing_results.get('recall') else "N/A")
                    with res_col3:
                        st.metric("F1 Score", f"{existing_results.get('f1_score', 0):.4f}" if existing_results.get('f1_score') else "N/A")
                    with res_col4:
                        st.metric("mAP@0.5", f"{existing_results.get('mAP50', 0):.4f}" if existing_results.get('mAP50') else "N/A")
                    with res_col5:
                        st.metric("mAP@0.5:0.95", f"{existing_results.get('AP5095', 0):.4f}" if existing_results.get('AP5095') else "N/A")
                
                st.markdown("---")
                
                # Dataset upload section
                st.markdown("### Upload Dataset (Roboflow Format)")
                with st.expander("📋 Dataset Format Instructions"):
                    st.markdown("""
                    **Roboflow Dataset Format:**
                    
                    Upload a `data.yaml` file from a Roboflow dataset. The file should contain:
                    
                    ```yaml
                    path: ../datasets/dataset_name
                    train: images/train
                    val: images/val
                    test: images/test
                    names:
                      0: class1
                      1: class2
                      2: class3
                    nc: 3
                    ```
                    
                    **Directory Structure:**
                    ```
                    dataset/
                    ├── data.yaml
                    ├── train/
                    │   ├── images/
                    │   └── labels/
                    ├── val/
                    │   ├── images/
                    │   └── labels/
                    └── test/
                        ├── images/
                        └── labels/
                    ```
                    
                    **Note:** 
                    - The validation dataset will be used for evaluation
                    - Images should be in `val/images/` directory
                    - Annotations should be in YOLO format in `val/labels/` directory
                    - Annotation files should have the same name as images but with `.txt` extension
                    """)
                
                st.info("Upload the data.yaml file from your Roboflow dataset. The validation split will be used for evaluation.")
                
                data_yaml_file = st.file_uploader(
                    "Upload data.yaml file",
                    type=["yaml", "yml"],
                    key="data_yaml_upload_eval"
                )
                
                if data_yaml_file:
                    if st.button("Run Evaluation", key="run_eval_button"):
                        with st.spinner("Loading validation dataset and running evaluation..."):
                            import tempfile
                            import shutil
                            
                            # Create temporary directory for dataset
                            temp_dir = tempfile.mkdtemp()
                            
                            try:
                                    # Save data.yaml file
                                    data_yaml_path = os.path.join(temp_dir, "data.yaml")
                                    with open(data_yaml_path, 'wb') as f:
                                        f.write(data_yaml_file.read())
                                    
                                    # Load validation dataset
                                    st.info("Loading validation dataset from data.yaml...")
                                    st.caption("⚠️ Note: The evaluation process is READ-ONLY. Your original dataset files will NOT be modified.")
                                    image_files, all_annotations, class_names = load_validation_dataset(data_yaml_path)
                                    
                                    if not image_files:
                                        st.error("No validation images found. Please check your data.yaml file and dataset structure.")
                                    else:
                                        st.success(f"Loaded {len(image_files)} validation images with annotations")
                                        
                                        # Flatten annotations for evaluation
                                        ground_truth = []
                                        for annotations in all_annotations:
                                            ground_truth.extend(annotations)
                                        
                                        st.info(f"Total ground truth annotations: {len(ground_truth)}")
                                        
                                        # Show ground truth class names from data.yaml
                                        if class_names:
                                            st.caption(f"📋 Ground truth classes from data.yaml: {sorted([class_names[i] for i in sorted(class_names.keys())])}")
                                        
                                        # Prepare model config for direct inference (bypassing API validation)
                                        model_config = {
                                            'model_id': selected_model_id,
                                            'model_name': selected_model.get('model_name', ''),
                                        }
                                        
                                        # Get preprocessing and postprocessing profiles for the selected model
                                        preprocess_profile = get_preprocess_for_model(selected_model)
                                        postprocess_profile_retrieved = get_postprocess_for_model(selected_model)
                                        
                                        # Prepare postprocess profile for solve_captcha (full structure)
                                        postprocess_profile = None
                                        if postprocess_profile_retrieved:
                                            postprocess_profile = {
                                                'postprocess_id': postprocess_profile_retrieved.get('postprocess_id'),
                                                'name': postprocess_profile_retrieved.get('name'),
                                                'steps': postprocess_profile_retrieved.get('steps', [])
                                            }
                                        
                                        # Run inference directly on validation images (no API, no validation)
                                        all_predictions = []
                                        
                                        progress_bar = st.progress(0)
                                        status_text = st.empty()
                                        
                                        for idx, img_path in enumerate(image_files):
                                            status_text.text(f"Processing image {idx + 1}/{len(image_files)}: {os.path.basename(img_path)}")
                                            progress_bar.progress((idx + 1) / len(image_files))
                                            
                                            try:
                                                # Read image file
                                                with open(img_path, 'rb') as f:
                                                    img_bytes = f.read()
                                                
                                                # Apply preprocessing if model has preprocess profile
                                                processed_img_bytes = img_bytes
                                                if preprocess_profile:
                                                    try:
                                                        processed_img_bytes, _ = apply_preprocess(img_bytes, preprocess_profile)
                                                    except Exception as e:
                                                        st.warning(f"Preprocessing failed for {os.path.basename(img_path)}: {e}")
                                                        processed_img_bytes = img_bytes
                                                
                                                # Run inference directly using solve_captcha (bypasses question/challenge_type validation)
                                                # Pass empty question since we're bypassing validation
                                                inference_result = solve_captcha(
                                                    processed_img_bytes,
                                                    question="",  # Empty question - bypasses validation
                                                    config=model_config,
                                                    postprocess_profile=postprocess_profile
                                                )
                                                
                                                # Handle inference result
                                                if isinstance(inference_result, dict):
                                                    if 'error' in inference_result:
                                                        st.warning(f"Inference error for {os.path.basename(img_path)}: {inference_result['error']}")
                                                        continue
                                                    elif 'message' in inference_result:
                                                        # No detections
                                                        detections = []
                                                    else:
                                                        detections = inference_result
                                                elif isinstance(inference_result, list):
                                                    detections = inference_result
                                                else:
                                                    detections = []
                                                
                                                # Add image identifier to each detection
                                                for det in detections:
                                                    if isinstance(det, dict):
                                                        det['image_id'] = os.path.basename(img_path)
                                                
                                                all_predictions.extend(detections)
                                            except Exception as e:
                                                st.warning(f"Failed to run inference on {os.path.basename(img_path)}: {e}")
                                                import traceback
                                                st.code(traceback.format_exc())
                                        
                                        progress_bar.empty()
                                        status_text.empty()
                                        
                                        if all_predictions:
                                            st.success(f"Generated {len(all_predictions)} predictions from {len(image_files)} validation images")
                                            
                                            # Debug: Show class names from predictions and ground truth
                                            pred_classes = set([p.get('class', '') for p in all_predictions if isinstance(p, dict)])
                                            gt_classes = set([g.get('class', '') for g in ground_truth if isinstance(g, dict)])
                                            
                                            with st.expander("🔍 Debug: Class Name Comparison"):
                                                st.write("**Prediction Classes:**", sorted(pred_classes))
                                                st.write("**Ground Truth Classes:**", sorted(gt_classes))
                                                st.write("**Matching Classes:**", sorted(pred_classes & gt_classes))
                                                st.write("**Only in Predictions:**", sorted(pred_classes - gt_classes))
                                                st.write("**Only in Ground Truth:**", sorted(gt_classes - pred_classes))
                                            
                                            # Create class mapping if needed (map model class names to ground truth class names)
                                            # This handles cases where model uses different class names than ground truth
                                            class_mapping = {}
                                            if pred_classes != gt_classes:
                                                st.warning("⚠️ Class name mismatch detected! Attempting automatic mapping...")
                                                # Try to match by similarity (case-insensitive, underscore/space normalization)
                                                for pred_class in pred_classes:
                                                    pred_normalized = pred_class.lower().replace('_', ' ').replace('-', ' ')
                                                    for gt_class in gt_classes:
                                                        gt_normalized = gt_class.lower().replace('_', ' ').replace('-', ' ')
                                                        if pred_normalized == gt_normalized:
                                                            class_mapping[pred_class] = gt_class
                                                            break
                                                
                                                if class_mapping:
                                                    st.info(f"✅ Created class mapping: {class_mapping}")
                                                else:
                                                    st.error("❌ Could not automatically map class names. Please ensure model and dataset use the same class names.")
                                            
                                            # Run evaluation with class mapping
                                            eval_results = evaluate_model(
                                                all_predictions,
                                                ground_truth,
                                                iou_threshold=0.5,
                                                class_mapping=class_mapping if class_mapping else None
                                            )
                                            
                                            # Display overall metrics
                                            st.markdown("### Overall Evaluation Metrics")
                                            overall = eval_results['overall_metrics']
                                            
                                            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
                                            with metric_col1:
                                                st.metric("mAP@0.5", f"{overall['map_50']:.4f}")
                                            with metric_col2:
                                                st.metric("mAP@0.5:0.95", f"{overall['map_50_95']:.4f}")
                                            with metric_col3:
                                                st.metric("Total TP", overall['total_tp'])
                                            with metric_col4:
                                                st.metric("Total FP", overall['total_fp'])
                                            with metric_col5:
                                                st.metric("Total FN", overall['total_fn'])
                                            
                                            # Calculate macro-averaged precision, recall, and F1 from per-class metrics
                                            per_class = eval_results['per_class_metrics']
                                            macro_precision = 0.0
                                            macro_recall = 0.0
                                            macro_f1 = 0.0
                                            if per_class:
                                                precisions = [m['precision'] for m in per_class.values()]
                                                recalls = [m['recall'] for m in per_class.values()]
                                                f1_scores = [m['f1_score'] for m in per_class.values()]
                                                macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
                                                macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
                                                macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
                                            
                                            # Prepare results for database update
                                            # Format matches expected structure: {precision, recall, f1_score, mAP50, AP5095}
                                            db_results = {
                                                'precision': macro_precision,
                                                'recall': macro_recall,
                                                'f1_score': macro_f1,
                                                'mAP50': overall['map_50'],
                                                'AP5095': overall['map_50_95'],
                                                'total_tp': overall['total_tp'],
                                                'total_fp': overall['total_fp'],
                                                'total_fn': overall['total_fn'],
                                                'evaluation_timestamp': datetime.now().isoformat(),
                                                'per_class_metrics': per_class  # Store detailed per-class metrics
                                            }
                                            
                                            # Update model in database with evaluation results
                                            try:
                                                updated_model = upsert_model(
                                                    model_id=selected_model_id,
                                                    model_name=selected_model.get('model_name', ''),
                                                    weights_file_stream=None,  # Don't update weights
                                                    results=db_results,
                                                    is_active=selected_model.get('is_active', False)  # Preserve is_active status
                                                )
                                                if updated_model:
                                                    st.success("✅ Evaluation results saved to database!")
                                                else:
                                                    st.warning("⚠️ Could not save evaluation results to database. Please check database connection.")
                                            except Exception as e:
                                                st.error(f"❌ Error saving evaluation results: {e}")
                                                import traceback
                                                st.code(traceback.format_exc())
                                            
                                            # Per-class metrics table
                                            st.markdown("### Per-Class Metrics")
                                            per_class = eval_results['per_class_metrics']
                                            
                                            if per_class:
                                                class_data = []
                                                for class_name, metrics in per_class.items():
                                                    class_data.append({
                                                        'Class': class_name,
                                                        'Precision': f"{metrics['precision']:.4f}",
                                                        'Recall': f"{metrics['recall']:.4f}",
                                                        'F1 Score': f"{metrics['f1_score']:.4f}",
                                                        'AP@0.5': f"{metrics['ap_50']:.4f}",
                                                        'TP': metrics['tp'],
                                                        'FP': metrics['fp'],
                                                        'FN': metrics['fn']
                                                    })
                                                
                                                df_metrics = pd.DataFrame(class_data)
                                                st.dataframe(df_metrics, width='stretch')
                                                
                                                # Per-class metrics charts
                                                st.markdown("#### Per-Class Metrics Visualization")
                                                
                                                if PLOTLY_AVAILABLE:
                                                    classes = list(per_class.keys())
                                                    precisions = [per_class[c]['precision'] for c in classes]
                                                    recalls = [per_class[c]['recall'] for c in classes]
                                                    f1_scores = [per_class[c]['f1_score'] for c in classes]
                                                    
                                                    # Precision, Recall, F1 chart
                                                    fig_prf = go.Figure()
                                                    fig_prf.add_trace(go.Bar(
                                                        name='Precision',
                                                        x=classes,
                                                        y=precisions,
                                                        marker_color='lightblue'
                                                    ))
                                                    fig_prf.add_trace(go.Bar(
                                                        name='Recall',
                                                        x=classes,
                                                        y=recalls,
                                                        marker_color='lightgreen'
                                                    ))
                                                    fig_prf.add_trace(go.Bar(
                                                        name='F1 Score',
                                                        x=classes,
                                                        y=f1_scores,
                                                        marker_color='lightcoral'
                                                    ))
                                                    fig_prf.update_layout(
                                                        title='Per-Class Precision, Recall, and F1 Score',
                                                        xaxis_title='Class',
                                                        yaxis_title='Score',
                                                        barmode='group',
                                                        height=400
                                                    )
                                                    st.plotly_chart(fig_prf, use_container_width=True)
                                                    
                                                    # AP@0.5 chart
                                                    fig_ap = go.Figure()
                                                    aps = [per_class[c]['ap_50'] for c in classes]
                                                    fig_ap.add_trace(go.Bar(
                                                        x=classes,
                                                        y=aps,
                                                        marker_color='steelblue'
                                                    ))
                                                    fig_ap.update_layout(
                                                        title='Per-Class AP@0.5',
                                                        xaxis_title='Class',
                                                        yaxis_title='AP@0.5',
                                                        height=400
                                                    )
                                                    st.plotly_chart(fig_ap, use_container_width=True)
                                                    
                                                    # TP, FP, FN chart
                                                    fig_counts = go.Figure()
                                                    tps = [per_class[c]['tp'] for c in classes]
                                                    fps = [per_class[c]['fp'] for c in classes]
                                                    fns = [per_class[c]['fn'] for c in classes]
                                                    fig_counts.add_trace(go.Bar(
                                                        name='True Positives',
                                                        x=classes,
                                                        y=tps,
                                                        marker_color='green'
                                                    ))
                                                    fig_counts.add_trace(go.Bar(
                                                        name='False Positives',
                                                        x=classes,
                                                        y=fps,
                                                        marker_color='red'
                                                    ))
                                                    fig_counts.add_trace(go.Bar(
                                                        name='False Negatives',
                                                        x=classes,
                                                        y=fns,
                                                        marker_color='orange'
                                                    ))
                                                    fig_counts.update_layout(
                                                        title='Per-Class Detection Counts (TP, FP, FN)',
                                                        xaxis_title='Class',
                                                        yaxis_title='Count',
                                                        barmode='group',
                                                        height=400
                                                    )
                                                    st.plotly_chart(fig_counts, use_container_width=True)
                                                    
                                                    # Overall metrics summary chart
                                                    st.markdown("#### Overall Metrics Summary")
                                                    summary_fig = go.Figure()
                                                    summary_fig.add_trace(go.Bar(
                                                        x=['mAP@0.5', 'mAP@0.5:0.95'],
                                                        y=[overall['map_50'], overall['map_50_95']],
                                                        marker_color='purple',
                                                        text=[f"{overall['map_50']:.4f}", f"{overall['map_50_95']:.4f}"],
                                                        textposition='auto'
                                                    ))
                                                    summary_fig.update_layout(
                                                        title='Overall mAP Metrics',
                                                        xaxis_title='Metric',
                                                        yaxis_title='Score',
                                                        height=400
                                                    )
                                                    st.plotly_chart(summary_fig, use_container_width=True)
                                                else:
                                                    st.info("📊 Install plotly to view interactive charts: `pip install plotly`")
                                            else:
                                                st.warning("No per-class metrics available.")
                                        else:
                                            st.error("No predictions generated. Please check your validation images and model configuration.")
                                
                            except Exception as e:
                                st.error(f"Evaluation failed: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                            finally:
                                # Cleanup temporary directory
                                if 'temp_dir' in locals():
                                    shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        import traceback
        st.code(traceback.format_exc())
    
elif main_section == "7. hCAPTCHA Demo":
    if demo_subsection == "7a. Upload Image for Inference":
        # Use existing "Upload -> API" functionality
        uploaded_img = st.file_uploader("Upload hCAPTCHA Image", type=["jpg", "jpeg", "png"])
        question = st.text_input("Enter hCAPTCHA Question")
        if st.button("Send to API"):
            if uploaded_img and question:
                status.info("Uploading...")
                time.sleep(0.2)
                progress.progress(20)
                status.info("Sending to API...")
                files = {"image": uploaded_img}
                data = {"question": question}
                resp = requests.post("http://localhost:5000/solve_hcaptcha", files=files, data=data)
                progress.progress(70)
                if resp.ok:
                    out = resp.json()
                    if out.get("message"):
                        st.warning(out.get("message"))
                    st.success("Uploaded and processed by API")
                    
                    # Display basic info
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Display processed image with bounding boxes
                        processed_img_b64 = out.get("processed_image")
                        results = out.get("results", [])
                        
                        if processed_img_b64:
                            try:
                                img_bytes = base64.b64decode(processed_img_b64)
                                img = Image.open(io.BytesIO(img_bytes))
                                
                                # Draw bounding boxes on image
                                if results:
                                    draw = ImageDraw.Draw(img)
                                    # Try to load a font, fallback to default if not available
                                    try:
                                        font = ImageFont.truetype("arial.ttf", 16)
                                    except:
                                        try:
                                            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                                        except:
                                            font = ImageFont.load_default()
                                    
                                    for idx, result in enumerate(results):
                                        bbox = result.get('bbox', [])
                                        if len(bbox) >= 4:
                                            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                                            # Draw rectangle
                                            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                                            # Draw label
                                            class_name = result.get('class', 'Unknown')
                                            confidence = result.get('confidence', 0.0)
                                            label = f"{class_name}: {confidence:.2f}"
                                            # Get text size for background
                                            bbox_text = draw.textbbox((0, 0), label, font=font)
                                            text_width = bbox_text[2] - bbox_text[0]
                                            text_height = bbox_text[3] - bbox_text[1]
                                            # Draw text background
                                            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill="red")
                                            # Draw text
                                            draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=font)
                                    
                                    st.image(img, caption="Processed Image with Detections", width='stretch')
                            except Exception as e:
                                st.error(f"Failed to display image: {e}")
                        else:
                            st.warning("No processed image available")
                    
                    with col2:
                        # Display preprocessing info
                        preprocess = out.get("preprocess")
                        if preprocess:
                            st.info(f"**Preprocessing:** {preprocess.get('preprocess_id', 'N/A')}")
                            if preprocess.get('applied_steps'):
                                st.caption(f"Steps: {', '.join([step.get('operation', '') for step in preprocess.get('applied_steps', [])])}")
                        
                        # Display postprocessing info
                        postprocess = out.get("postprocess")
                        if postprocess:
                            steps = postprocess.get('steps', [])
                            # Extract thresholds from nms operation if present
                            conf_thresh = 'N/A'
                            iou_thresh = 'N/A'
                            if isinstance(steps, list):
                                for step in steps:
                                    if isinstance(step, dict) and step.get('operation') == 'nms':
                                        params = step.get('params', {})
                                        conf_thresh = params.get('confidence_threshold', 'N/A')
                                        iou_thresh = params.get('iou_threshold', 'N/A')
                                        break
                            elif isinstance(steps, dict):
                                # Legacy format support
                                conf_thresh = steps.get('confidence_threshold', 'N/A')
                                iou_thresh = steps.get('iou_threshold', 'N/A')
                            
                            st.info(f"**Postprocessing:** {postprocess.get('name', 'N/A')}")
                            if isinstance(steps, list) and len(steps) > 0:
                                operations = [step.get('operation', 'Unknown') for step in steps if isinstance(step, dict)]
                                st.caption(f"Operations: {', '.join(operations)}")
                                if conf_thresh != 'N/A' or iou_thresh != 'N/A':
                                    st.caption(f"Conf: {conf_thresh}, IoU: {iou_thresh}")
                            else:
                                st.caption(f"Conf: {conf_thresh}, IoU: {iou_thresh}")
                        
                        # Display performance
                        st.metric("Processing Time", f"{out.get('perform_time', 0):.3f}s")
                    
                    # Display results table
                    if results:
                        st.markdown("### Detection Results")
                        # Prepare data for table
                        table_data = []
                        for idx, result in enumerate(results):
                            bbox = result.get('bbox', [])
                            bbox_str = f"[{', '.join([str(int(x)) for x in bbox[:4]])}]" if len(bbox) >= 4 else "N/A"
                            table_data.append({
                                "ID": idx + 1,
                                "Class": result.get('class', 'Unknown'),
                                "Confidence": f"{result.get('confidence', 0.0):.4f}",
                                "Bounding Box": bbox_str,
                                "Coordinates": f"({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})" if len(bbox) >= 4 else "N/A"
                            })
                        st.dataframe(table_data, width='stretch')
                    else:
                        st.info("No detections found")
                    
                    # Display additional metadata
                    with st.expander("View Full Response"):
                        st.json(out)
                    
                    progress.progress(100)
                else:
                    st.error("API request failed")
                    progress.progress(0)
            else:
                st.warning("Please upload an image and enter a question.")

    elif demo_subsection == "7b. Auto Crawler, Solver, and Clicker":
        # Use existing "Crawl -> API" functionality
        if st.button("Crawl Demo Site And Send To API", key="crawl_demo_button"):
            status.info("Launching browser and crawling demo site...")
            progress.progress(10)
            start = time.time()
            summary = run_crawl_once()
            elapsed = time.time() - start
            progress.progress(90)
            if summary.get("total_sent", 0) > 0:
                st.success(
                    f"Crawled. Question: '{summary.get('question')}'. "
                    f"Sent {summary.get('total_sent')} image(s) (canvas: {summary.get('sent_canvas')}, divs: {summary.get('sent_divs')})."
                )
                accepted = summary.get("accepted", []) or []
                if accepted:
                    # Check if this is a multi-crumb challenge (multiple challenges with different challenge_ids)
                    challenge_ids = set()
                    for item in accepted:
                        result = item.get("result", {})
                        if isinstance(result, dict) and result.get("challenge_id"):
                            challenge_ids.add(result.get("challenge_id"))
                    
                    is_multi_crumb = len(challenge_ids) > 1
                    
                    # Check if this is a batch result (multiple images with same result)
                    first_result = accepted[0].get("result", {}) if accepted else {}
                    is_batch = isinstance(first_result.get("results"), list) and len(first_result.get("results", [])) > 0 and isinstance(first_result.get("results", [])[0], dict) and "image_index" in first_result.get("results", [])[0]
                    
                    if is_multi_crumb:
                        # Display multi-crumb challenge results
                        st.markdown("### Multi-Crumb Challenge Results")
                        st.info(f"Detected {len(challenge_ids)} separate challenge(s) from multi-crumb challenge")
                        
                        # Group accepted results by challenge_id
                        challenges_by_id = {}
                        for item in accepted:
                            result = item.get("result", {})
                            challenge_id = result.get("challenge_id") if isinstance(result, dict) else None
                            if challenge_id:
                                if challenge_id not in challenges_by_id:
                                    challenges_by_id[challenge_id] = []
                                challenges_by_id[challenge_id].append(item)
                        
                        # Display each challenge separately
                        for challenge_idx, (challenge_id, challenge_items) in enumerate(challenges_by_id.items(), 1):
                            st.markdown(f"#### Challenge {challenge_idx} (ID: {challenge_id})")
                            
                            # Process each item in this challenge
                            for item_idx, item in enumerate(challenge_items):
                                result = item.get("result", {})
                                data_url = item.get("data_url")
                                filename = item.get("filename", f"image_{item_idx+1}.png")
                                
                                # Display image and results similar to single canvas flow
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    processed_img_b64 = result.get("processed_image") if isinstance(result, dict) else None
                                    results = result.get("results", []) if isinstance(result, dict) else []
                                    
                                    # Always show processed image if available, otherwise fallback to data_url
                                    display_img_b64 = None
                                    if processed_img_b64:
                                        display_img_b64 = processed_img_b64
                                    elif data_url:
                                        # Extract base64 from data_url if no processed_image
                                        try:
                                            b64_part = data_url.split(",", 1)[1] if "," in data_url else data_url
                                            display_img_b64 = b64_part
                                        except:
                                            pass
                                    
                                    if display_img_b64:
                                        try:
                                            img_bytes = base64.b64decode(display_img_b64)
                                            img = Image.open(io.BytesIO(img_bytes))
                                            
                                            # Draw bounding boxes if there are valid detections
                                            valid_detections = []
                                            if results and isinstance(results, list) and len(results) > 0:
                                                # Filter for valid detections (must have bbox with 4 elements)
                                                valid_detections = [
                                                    r for r in results 
                                                    if isinstance(r, dict) 
                                                    and 'error' not in r 
                                                    and 'bbox' in r 
                                                    and isinstance(r.get('bbox'), (list, tuple))
                                                    and len(r.get('bbox', [])) >= 4
                                                    and all(isinstance(x, (int, float)) for x in r.get('bbox', [])[:4])
                                                ]
                                                
                                                if valid_detections:
                                                    draw = ImageDraw.Draw(img)
                                                    try:
                                                        font = ImageFont.truetype("arial.ttf", 16)
                                                    except:
                                                        try:
                                                            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                                                        except:
                                                            font = ImageFont.load_default()
                                                    
                                                    for result_item in valid_detections:
                                                        bbox = result_item.get('bbox', [])
                                                        if len(bbox) >= 4:
                                                            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                                                            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                                                            class_name = result_item.get('class', 'Unknown')
                                                            confidence = result_item.get('confidence', 0.0)
                                                            label = f"{class_name}: {confidence:.2f}"
                                                            bbox_text = draw.textbbox((0, 0), label, font=font)
                                                            text_width = bbox_text[2] - bbox_text[0]
                                                            text_height = bbox_text[3] - bbox_text[1]
                                                            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill="red")
                                                            draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=font)
                                            
                                            # Always display the image (with or without detections)
                                            if valid_detections:
                                                st.image(img, caption=f"{filename} (Challenge {challenge_idx}) - With Detections", width='stretch')
                                            else:
                                                st.image(img, caption=f"{filename} (Challenge {challenge_idx}) - No Detections", width='stretch')
                                        except Exception as e:
                                            st.error(f"Failed to display image: {e}")
                                    else:
                                        # Fallback: try to display from data_url if no processed_image
                                        if data_url:
                                            try:
                                                b64_part = data_url.split(",", 1)[1] if "," in data_url else data_url
                                                img_bytes = base64.b64decode(b64_part)
                                                st.image(img_bytes, caption=f"{filename} (Challenge {challenge_idx}) - Original Image", width='stretch')
                                            except Exception:
                                                st.write(f"{filename} (Challenge {challenge_idx})")
                                        else:
                                            st.warning(f"No image data available for {filename}")
                                
                                with col2:
                                    if isinstance(result, dict):
                                        preprocess = result.get("preprocess")
                                        if preprocess:
                                            st.info(f"**Preprocessing:** {preprocess.get('preprocess_id', 'N/A')}")
                                        
                                        postprocess = result.get("postprocess")
                                        if postprocess:
                                            steps = postprocess.get('steps', [])
                                            # Extract thresholds from nms operation if present
                                            conf_thresh = 'N/A'
                                            iou_thresh = 'N/A'
                                            if isinstance(steps, list):
                                                for step in steps:
                                                    if isinstance(step, dict) and step.get('operation') == 'nms':
                                                        params = step.get('params', {})
                                                        conf_thresh = params.get('confidence_threshold', 'N/A')
                                                        iou_thresh = params.get('iou_threshold', 'N/A')
                                                        break
                                            elif isinstance(steps, dict):
                                                # Legacy format support
                                                conf_thresh = steps.get('confidence_threshold', 'N/A')
                                                iou_thresh = steps.get('iou_threshold', 'N/A')
                                            
                                            st.info(f"**Postprocessing:** {postprocess.get('name', 'N/A')}")
                                            if isinstance(steps, list) and len(steps) > 0:
                                                operations = [step.get('operation', 'Unknown') for step in steps if isinstance(step, dict)]
                                                st.caption(f"Operations: {', '.join(operations)}")
                                                if conf_thresh != 'N/A' or iou_thresh != 'N/A':
                                                    st.caption(f"Conf: {conf_thresh}, IoU: {iou_thresh}")
                                            else:
                                                st.caption(f"Conf: {conf_thresh}, IoU: {iou_thresh}")
                                        
                                        if 'perform_time' in result:
                                            st.metric("Processing Time", f"{result.get('perform_time', 0):.3f}s")
                                        
                                        if 'challenge_id' in result:
                                            st.caption(f"**Challenge ID:** {result.get('challenge_id')}")
                                        
                                        if 'error' in result:
                                            st.error(f"**Error:** {result.get('error')}")
                                            st.caption("X Inference NOT saved (errors are not stored)")
                                        elif results:
                                            if isinstance(results, list):
                                                # Filter for valid detections
                                                valid_detections = [
                                                    r for r in results 
                                                    if isinstance(r, dict) 
                                                    and 'error' not in r 
                                                    and 'bbox' in r 
                                                    and isinstance(r.get('bbox'), (list, tuple))
                                                    and len(r.get('bbox', [])) >= 4
                                                    and all(isinstance(x, (int, float)) for x in r.get('bbox', [])[:4])
                                                ]
                                                
                                                if len(valid_detections) > 0:
                                                    st.success(f"**Detections:** {len(valid_detections)} objects")
                                                    st.caption("/ Inference saved to database")
                                                else:
                                                    if len(results) > 0:
                                                        st.info(f"**No valid detections found** ({len(results)} invalid result(s))")
                                                    else:
                                                        st.info("**No detections found**")
                                                    st.caption("/ Inference saved to database")
                                            else:
                                                st.info("**No detections found**")
                                                st.caption("/ Inference saved to database")
                                        else:
                                            st.info("**No detections found**")
                                            st.caption("/ Inference saved to database")
                                
                                # Display results table only if there are valid detections
                                valid_detections_for_table = []
                                if results and isinstance(results, list) and len(results) > 0:
                                    valid_detections_for_table = [
                                        r for r in results 
                                        if isinstance(r, dict) 
                                        and 'error' not in r 
                                        and 'bbox' in r 
                                        and isinstance(r.get('bbox'), (list, tuple))
                                        and len(r.get('bbox', [])) >= 4
                                        and all(isinstance(x, (int, float)) for x in r.get('bbox', [])[:4])
                                    ]
                                
                                if valid_detections_for_table and len(valid_detections_for_table) > 0:
                                    st.markdown(f"**Detection Results for {filename}:**")
                                    table_data = []
                                    for result_idx, result_item in enumerate(valid_detections_for_table):
                                        bbox = result_item.get('bbox', [])
                                        bbox_str = f"[{', '.join([str(int(x)) for x in bbox[:4]])}]" if len(bbox) >= 4 else "N/A"
                                        table_data.append({
                                            "ID": result_idx + 1,
                                            "Class": result_item.get('class', 'Unknown'),
                                            "Confidence": f"{result_item.get('confidence', 0.0):.4f}",
                                            "Bounding Box": bbox_str,
                                            "Coordinates": f"({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})" if len(bbox) >= 4 else "N/A"
                                        })
                                    if table_data:
                                        st.dataframe(table_data, width='stretch')
                                else:
                                    # Show clear message when no detections found
                                    if results and isinstance(results, list) and len(results) > 0:
                                        st.info(f"**No valid detections found for {filename}** ({len(results)} invalid result(s))")
                                    else:
                                        st.info(f"**No detections found for {filename}**")
                                
                                if item_idx < len(challenge_items) - 1:
                                    st.markdown("---")
                            
                            if challenge_idx < len(challenges_by_id):
                                st.markdown("---")
                                st.markdown("---")
                    
                    elif is_batch:
                        # Display batch results similar to upload-API
                        st.markdown("### Batch Inference Results")
                        
                        # Display summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Images Processed", len(accepted))
                        with col2:
                            if isinstance(first_result, dict) and 'perform_time' in first_result:
                                st.metric("Processing Time", f"{first_result.get('perform_time', 0):.3f}s")
                        with col3:
                            if isinstance(first_result, dict) and 'challenge_id' in first_result:
                                st.metric("Challenge ID", first_result.get('challenge_id', 'N/A'))
                        
                        # Display preprocessing/postprocessing info
                        if isinstance(first_result, dict):
                            col1, col2 = st.columns(2)
                            with col1:
                                preprocess = first_result.get('preprocess')
                                if preprocess:
                                    st.info(f"**Preprocessing:** {preprocess.get('preprocess_id', 'N/A')}")
                                    if preprocess.get('name'):
                                        st.caption(f"Name: {preprocess.get('name')}")
                            
                            with col2:
                                postprocess = first_result.get('postprocess')
                                if postprocess:
                                    steps = postprocess.get('steps', [])
                                    # Extract thresholds from nms operation if present
                                    conf_thresh = 'N/A'
                                    iou_thresh = 'N/A'
                                    if isinstance(steps, list):
                                        for step in steps:
                                            if isinstance(step, dict) and step.get('operation') == 'nms':
                                                params = step.get('params', {})
                                                conf_thresh = params.get('confidence_threshold', 'N/A')
                                                iou_thresh = params.get('iou_threshold', 'N/A')
                                                break
                                    elif isinstance(steps, dict):
                                        # Legacy format support
                                        conf_thresh = steps.get('confidence_threshold', 'N/A')
                                        iou_thresh = steps.get('iou_threshold', 'N/A')
                                    
                                    st.info(f"**Postprocessing:** {postprocess.get('name', 'N/A')}")
                                    if isinstance(steps, list) and len(steps) > 0:
                                        operations = [step.get('operation', 'Unknown') for step in steps if isinstance(step, dict)]
                                        st.caption(f"Operations: {', '.join(operations)}")
                                        if conf_thresh != 'N/A' or iou_thresh != 'N/A':
                                            st.caption(f"Conf: {conf_thresh}, IoU: {iou_thresh}")
                                    else:
                                        st.caption(f"Conf: {conf_thresh}, IoU: {iou_thresh}")
                        
                        # Display each image with results
                        # For batch results, the 'results' field contains an array of batch items
                        # Each batch item has: {image_index, image_name, results: [detections]}
                        # IMPORTANT: batch_results should come from a batch item, not the sample tile
                        # All batch images share the same result object, so we can get it from any batch item
                        batch_results = []
                        processed_images = []
                        has_sample_tile = False
                        
                        # Detect if first item is a sample tile (sent separately, not in batch)
                        if len(accepted) > 1:
                            first_item_result = accepted[0].get("result", {})
                            batch_item_result = accepted[1].get("result", {}) if len(accepted) > 1 else {}
                            
                            # Check if first item is a sample tile
                            if isinstance(first_item_result, dict):
                                first_results = first_item_result.get('results', [])
                                if isinstance(first_results, list) and len(first_results) > 0:
                                    if not isinstance(first_results[0], dict) or 'image_index' not in first_results[0]:
                                        has_sample_tile = True
                                elif isinstance(first_item_result, dict) and 'processed_image' in first_item_result:
                                    has_sample_tile = True
                            
                            # Get batch_results from a batch item (not the sample tile)
                            if has_sample_tile and isinstance(batch_item_result, dict):
                                batch_results = batch_item_result.get('results', [])
                                processed_images = batch_item_result.get('processed_images', [])
                            elif not has_sample_tile and isinstance(first_result, dict):
                                # No sample tile, batch results are in first_result
                                batch_results = first_result.get('results', [])
                                processed_images = first_result.get('processed_images', [])
                        
                        # Debug: Log batch results structure
                        print(f"Debug: batch_results type={type(batch_results)}, length={len(batch_results) if isinstance(batch_results, list) else 'N/A'}")
                        if batch_results and len(batch_results) > 0:
                            print(f"Debug: First batch item keys: {list(batch_results[0].keys()) if isinstance(batch_results[0], dict) else 'N/A'}")
                            if isinstance(batch_results[0], dict):
                                print(f"Debug: First batch item image_index: {batch_results[0].get('image_index')}")
                                print(f"Debug: First batch item results type: {type(batch_results[0].get('results'))}, length={len(batch_results[0].get('results', [])) if isinstance(batch_results[0].get('results'), list) else 'N/A'}")
                        
                        # Create a mapping from image_index to batch result for easier lookup
                        batch_results_by_index = {}
                        for batch_item in batch_results:
                            if isinstance(batch_item, dict) and 'image_index' in batch_item:
                                batch_results_by_index[batch_item['image_index']] = batch_item
                        
                        print(f"Debug: batch_results_by_index keys: {list(batch_results_by_index.keys())}")
                        
                        sample_offset = 1 if has_sample_tile else 0
                        
                        for img_idx, item in enumerate(accepted):
                            data_url = item.get("data_url")
                            result = item.get("result", {})
                            filename = item.get("filename", f"image_{img_idx+1}.png")
                            
                            if has_sample_tile and img_idx == 0:
                                image_result_data = None
                            else:
                                batch_idx = img_idx - sample_offset
                                image_index = batch_idx + 1
                                # Try to get result by image_index first
                                # Note: image_index in API is 1-based (1, 2, 3, ...)
                                image_result_data = batch_results_by_index.get(image_index)
                                
                                # Fallback: try by batch index if not found by image_index
                                if image_result_data is None and batch_idx >= 0 and batch_idx < len(batch_results):
                                    image_result_data = batch_results[batch_idx]
                                    print(f"Debug: Using fallback - found result by batch_idx={batch_idx} for image_index={image_index}")
                                
                                # Debug: Log result retrieval
                                if image_result_data:
                                    img_res = image_result_data.get('results', [])
                                    print(f"Debug: Found image_result_data for {filename}: image_index={image_result_data.get('image_index')}, results type={type(img_res)}, results length={len(img_res) if isinstance(img_res, list) else 'N/A'}")
                                    if isinstance(img_res, list) and len(img_res) > 0:
                                        print(f"Debug: First result keys: {list(img_res[0].keys()) if isinstance(img_res[0], dict) else 'N/A'}")
                                        if isinstance(img_res[0], dict) and 'bbox' in img_res[0]:
                                            print(f"Debug: First result bbox: {img_res[0].get('bbox')}, bbox type: {type(img_res[0].get('bbox'))}, bbox length: {len(img_res[0].get('bbox', [])) if isinstance(img_res[0].get('bbox'), (list, tuple)) else 'N/A'}")
                                else:
                                    print(f"Debug: Could not find result for img_idx={img_idx}, batch_idx={batch_idx}, image_index={image_index}")
                                    print(f"Debug: batch_results_by_index keys: {list(batch_results_by_index.keys())}")
                                    print(f"Debug: batch_results length: {len(batch_results)}")
                                    if batch_results and len(batch_results) > 0:
                                        print(f"Debug: First batch result keys: {list(batch_results[0].keys()) if isinstance(batch_results[0], dict) else 'N/A'}")
                                        if isinstance(batch_results[0], dict):
                                            print(f"Debug: First batch result image_index: {batch_results[0].get('image_index')}")
                            
                            st.markdown(f"#### {filename}")
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                display_img_b64 = None
                                if has_sample_tile and img_idx == 0:
                                    # For sample tile, prefer processed_image, fallback to data_url
                                    if isinstance(result, dict) and result.get('processed_image'):
                                        display_img_b64 = result.get('processed_image')
                                    elif data_url:
                                        # Extract base64 from data_url if no processed_image
                                        try:
                                            b64_part = data_url.split(",", 1)[1] if "," in data_url else data_url
                                            display_img_b64 = b64_part
                                        except:
                                            pass
                                elif processed_images:
                                    batch_img_idx = img_idx - sample_offset
                                    if batch_img_idx >= 0 and batch_img_idx < len(processed_images):
                                        display_img_b64 = processed_images[batch_img_idx]
                                
                                if display_img_b64:
                                    try:
                                        img_bytes = base64.b64decode(display_img_b64)
                                        img = Image.open(io.BytesIO(img_bytes))
                                        
                                        results_to_draw = None
                                        if has_sample_tile and img_idx == 0:
                                            results_to_draw = result.get('results', []) if isinstance(result, dict) else []
                                        elif image_result_data:
                                            results_to_draw = image_result_data.get('results', [])
                                        
                                        if results_to_draw and isinstance(results_to_draw, list) and len(results_to_draw) > 0:
                                            valid_detections = [r for r in results_to_draw if isinstance(r, dict) and 'error' not in r and 'bbox' in r and len(r.get('bbox', [])) >= 4]
                                            if valid_detections:
                                                draw = ImageDraw.Draw(img)
                                                try:
                                                    font = ImageFont.truetype("arial.ttf", 16)
                                                except:
                                                    try:
                                                        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                                                    except:
                                                        font = ImageFont.load_default()
                                                
                                                for det_result in valid_detections:
                                                    bbox = det_result.get('bbox', [])
                                                    if len(bbox) >= 4:
                                                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                                                        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                                                        class_name = det_result.get('class', 'Unknown')
                                                        confidence = det_result.get('confidence', 0.0)
                                                        label = f"{class_name}: {confidence:.2f}"
                                                        bbox_text = draw.textbbox((0, 0), label, font=font)
                                                        text_width = bbox_text[2] - bbox_text[0]
                                                        text_height = bbox_text[3] - bbox_text[1]
                                                        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill="red")
                                                        draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=font)
                                        
                                        st.image(img, caption=f"{filename} (with detections)", width='stretch')
                                    except Exception as e:
                                        st.error(f"Failed to display image: {e}")
                                else:
                                    if data_url:
                                        try:
                                            b64_part = data_url.split(",", 1)[1] if "," in data_url else data_url
                                            img_bytes = base64.b64decode(b64_part)
                                            st.image(img_bytes, caption=f"{filename}", width='stretch')
                                        except Exception:
                                            st.write(f"{filename}")
                            
                            with col2:
                                display_result = None
                                if has_sample_tile and img_idx == 0:
                                    display_result = result
                                elif image_result_data:
                                    display_result = image_result_data
                                
                                if display_result:
                                    img_results = display_result.get('results', [])
                                    if isinstance(img_results, dict) and 'error' in img_results:
                                        st.error(f"**Error:** {img_results.get('error')}")
                                        st.caption("X Inference NOT saved (errors are not stored)")
                                    elif isinstance(img_results, list):
                                        if len(img_results) > 0:
                                            st.success(f"**Detections:** {len(img_results)} objects")
                                        else:
                                            st.info("No detections found")
                                        st.caption("/ Inference saved to database")
                                    else:
                                        st.info("No detections found")
                                        st.caption("/ Inference saved to database")
                                else:
                                    st.warning("No result data available for this image")
                                
                                if isinstance(result, dict) and result.get('model'):
                                    model = result.get('model', {})
                                    st.caption(f"**Model:** {model.get('model_name', 'Unknown')}")
                                
                                table_display_result = None
                                if has_sample_tile and img_idx == 0:
                                    # For sample tile, use the result directly (single image API response)
                                    table_display_result = result
                                elif image_result_data:
                                    # For batch tiles, use the individual image result from batch
                                    table_display_result = image_result_data
                                
                                if table_display_result:
                                    # Extract results from the appropriate structure
                                    # For batch items: table_display_result is {image_index, image_name, results: [detections]}
                                    # For single image: table_display_result is {results: [detections], ...}
                                    img_results = table_display_result.get('results', [])
                                    
                                    # Debug: Log the results structure
                                    print(f"Debug: table_display_result for {filename}: type={type(img_results)}, length={len(img_results) if isinstance(img_results, list) else 'N/A'}")
                                    if isinstance(img_results, list) and len(img_results) > 0:
                                        print(f"Debug: First result item type: {type(img_results[0])}, keys: {list(img_results[0].keys()) if isinstance(img_results[0], dict) else 'N/A'}")
                                    
                                    if isinstance(img_results, list):
                                        if len(img_results) > 0:
                                            # Filter for valid detections (must have bbox with 4 elements)
                                            # Only include detections that have a valid bbox with at least 4 elements
                                            valid_detections = [
                                                r for r in img_results 
                                                if isinstance(r, dict) 
                                                and 'error' not in r 
                                                and 'bbox' in r 
                                                and isinstance(r.get('bbox'), (list, tuple))
                                                and len(r.get('bbox', [])) >= 4
                                                and all(isinstance(x, (int, float)) for x in r.get('bbox', [])[:4])
                                            ]
                                            
                                            if valid_detections:
                                                st.markdown(f"**Detection Results for {filename}:**")
                                                table_data = []
                                                for det_idx, det_result in enumerate(valid_detections):
                                                    bbox = det_result.get('bbox', [])
                                                    bbox_str = f"[{', '.join([str(int(x)) for x in bbox[:4]])}]" if len(bbox) >= 4 else "N/A"
                                                    table_data.append({
                                                        "ID": det_idx + 1,
                                                        "Class": det_result.get('class', 'Unknown'),
                                                        "Confidence": f"{det_result.get('confidence', 0.0):.4f}",
                                                        "Bounding Box": bbox_str,
                                                        "Coordinates": f"({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})" if len(bbox) >= 4 else "N/A"
                                                    })
                                                if table_data:
                                                    st.dataframe(table_data, width='stretch')
                                            else:
                                                # No valid detections - check if there are any results at all
                                                if len(img_results) > 0:
                                                    # There are results but they're invalid - show message
                                                    st.info(f"No valid detections for {filename} ({len(img_results)} invalid result(s))")
                                                else:
                                                    # Empty results - no detections
                                                    st.info(f"No detections found for {filename}")
                                        else:
                                            # Empty list - no detections
                                            st.info(f"No detections found for {filename}")
                                    elif isinstance(img_results, dict):
                                        if 'error' in img_results:
                                            st.error(f"Error: {img_results.get('error')}")
                                        elif 'message' in img_results:
                                            st.info(f"Message: {img_results.get('message')}")
                                        else:
                                            st.warning(f"Unexpected result format for {filename}")
                                    else:
                                        st.info(f"No detections found for {filename} (results type: {type(img_results)})")
                                else:
                                    st.info(f"No result data available for {filename}")
                            
                            st.markdown("---")
                        
                        with st.expander("View Full Response"):
                            st.json(first_result)
                
                    else:
                        # Display single canvas results
                        for idx, item in enumerate(accepted):
                            data_url = item.get("data_url")
                            result = item.get("result", {})
                            filename = item.get("filename", f"image_{idx+1}.png")
                            
                            st.markdown(f"### {filename}")
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                processed_img_b64 = result.get("processed_image") if isinstance(result, dict) else None
                                results = result.get("results", []) if isinstance(result, dict) else []
                                
                                # Always show processed image if available, otherwise fallback to data_url
                                display_img_b64 = None
                                if processed_img_b64:
                                    display_img_b64 = processed_img_b64
                                elif data_url:
                                    # Extract base64 from data_url if no processed_image
                                    try:
                                        b64_part = data_url.split(",", 1)[1] if "," in data_url else data_url
                                        display_img_b64 = b64_part
                                    except:
                                        pass
                                
                                if display_img_b64:
                                    try:
                                        img_bytes = base64.b64decode(display_img_b64)
                                        img = Image.open(io.BytesIO(img_bytes))
                                        
                                        # Draw bounding boxes if there are valid detections
                                        valid_detections = []
                                        if results and isinstance(results, list) and len(results) > 0:
                                            # Filter for valid detections (must have bbox with 4 elements)
                                            valid_detections = [
                                                r for r in results 
                                                if isinstance(r, dict) 
                                                and 'error' not in r 
                                                and 'bbox' in r 
                                                and isinstance(r.get('bbox'), (list, tuple))
                                                and len(r.get('bbox', [])) >= 4
                                                and all(isinstance(x, (int, float)) for x in r.get('bbox', [])[:4])
                                            ]
                                            
                                            if valid_detections:
                                                draw = ImageDraw.Draw(img)
                                                try:
                                                    font = ImageFont.truetype("arial.ttf", 16)
                                                except:
                                                    try:
                                                        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                                                    except:
                                                        font = ImageFont.load_default()
                                                
                                                for result_item in valid_detections:
                                                    bbox = result_item.get('bbox', [])
                                                    if len(bbox) >= 4:
                                                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                                                        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                                                        class_name = result_item.get('class', 'Unknown')
                                                        confidence = result_item.get('confidence', 0.0)
                                                        label = f"{class_name}: {confidence:.2f}"
                                                        bbox_text = draw.textbbox((0, 0), label, font=font)
                                                        text_width = bbox_text[2] - bbox_text[0]
                                                        text_height = bbox_text[3] - bbox_text[1]
                                                        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill="red")
                                                        draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=font)
                                        
                                        # Always display the image (with or without detections)
                                        if valid_detections:
                                            st.image(img, caption="Processed Image with Detections", width='stretch')
                                        else:
                                            st.image(img, caption="Processed Image (No Detections)", width='stretch')
                                    except Exception as e:
                                        st.error(f"Failed to display image: {e}")
                                else:
                                    if data_url:
                                        try:
                                            b64_part = data_url.split(",", 1)[1] if "," in data_url else data_url
                                            img_bytes = base64.b64decode(b64_part)
                                            st.image(img_bytes, caption=f"{filename}", width='stretch')
                                        except Exception:
                                            st.write(f"{filename}")
                            
                            with col2:
                                if isinstance(result, dict):
                                    preprocess = result.get("preprocess")
                                    if preprocess:
                                        st.info(f"**Preprocessing:** {preprocess.get('preprocess_id', 'N/A')}")
                                        if preprocess.get('applied_steps'):
                                            st.caption(f"Steps: {', '.join([step.get('operation', '') for step in preprocess.get('applied_steps', [])])}")
                                    
                                    postprocess = result.get("postprocess")
                                    if postprocess:
                                        steps = postprocess.get('steps', [])
                                        # Extract thresholds from nms operation if present
                                        conf_thresh = 'N/A'
                                        iou_thresh = 'N/A'
                                        if isinstance(steps, list):
                                            for step in steps:
                                                if isinstance(step, dict) and step.get('operation') == 'nms':
                                                    params = step.get('params', {})
                                                    conf_thresh = params.get('confidence_threshold', 'N/A')
                                                    iou_thresh = params.get('iou_threshold', 'N/A')
                                                    break
                                        elif isinstance(steps, dict):
                                            # Legacy format support
                                            conf_thresh = steps.get('confidence_threshold', 'N/A')
                                            iou_thresh = steps.get('iou_threshold', 'N/A')
                                        
                                        st.info(f"**Postprocessing:** {postprocess.get('name', 'N/A')}")
                                        if isinstance(steps, list) and len(steps) > 0:
                                            operations = [step.get('operation', 'Unknown') for step in steps if isinstance(step, dict)]
                                            st.caption(f"Operations: {', '.join(operations)}")
                                            if conf_thresh != 'N/A' or iou_thresh != 'N/A':
                                                st.caption(f"Conf: {conf_thresh}, IoU: {iou_thresh}")
                                        else:
                                            st.caption(f"Conf: {conf_thresh}, IoU: {iou_thresh}")
                                    
                                    if 'perform_time' in result:
                                        st.metric("Processing Time", f"{result.get('perform_time', 0):.3f}s")
                                    
                                    if 'challenge_id' in result:
                                        st.caption(f"**Challenge ID:** {result.get('challenge_id')}")
                                    
                                    if 'error' in result:
                                        st.error(f"**Error:** {result.get('error')}")
                                        st.caption("X Inference NOT saved (errors are not stored)")
                                    elif results:
                                        if isinstance(results, list):
                                            # Filter for valid detections
                                            valid_detections = [
                                                r for r in results 
                                                if isinstance(r, dict) 
                                                and 'error' not in r 
                                                and 'bbox' in r 
                                                and isinstance(r.get('bbox'), (list, tuple))
                                                and len(r.get('bbox', [])) >= 4
                                                and all(isinstance(x, (int, float)) for x in r.get('bbox', [])[:4])
                                            ]
                                            
                                            if len(valid_detections) > 0:
                                                st.success(f"**Detections:** {len(valid_detections)} objects")
                                                st.caption("/ Inference saved to database")
                                            else:
                                                if len(results) > 0:
                                                    st.info(f"**No valid detections found** ({len(results)} invalid result(s))")
                                                else:
                                                    st.info("**No detections found**")
                                                st.caption("/ Inference saved to database")
                                        elif isinstance(results, dict) and 'message' in results:
                                            st.info(f"**Message:** {results.get('message')}")
                                            st.caption("/ Inference saved to database")
                                        else:
                                            st.info("**No detections found**")
                                            st.caption("/ Inference saved to database")
                                    else:
                                        st.info("**No detections found**")
                                        st.caption("/ Inference saved to database")
                            
                            # Display results table only if there are valid detections
                            valid_detections_for_table = []
                            if results and isinstance(results, list) and len(results) > 0:
                                valid_detections_for_table = [
                                    r for r in results 
                                    if isinstance(r, dict) 
                                    and 'error' not in r 
                                    and 'bbox' in r 
                                    and isinstance(r.get('bbox'), (list, tuple))
                                    and len(r.get('bbox', [])) >= 4
                                    and all(isinstance(x, (int, float)) for x in r.get('bbox', [])[:4])
                                ]
                            
                            if valid_detections_for_table and len(valid_detections_for_table) > 0:
                                st.markdown("### Detection Results")
                                table_data = []
                                for result_idx, result_item in enumerate(valid_detections_for_table):
                                    bbox = result_item.get('bbox', [])
                                    bbox_str = f"[{', '.join([str(int(x)) for x in bbox[:4]])}]" if len(bbox) >= 4 else "N/A"
                                    table_data.append({
                                        "ID": result_idx + 1,
                                        "Class": result_item.get('class', 'Unknown'),
                                        "Confidence": f"{result_item.get('confidence', 0.0):.4f}",
                                        "Bounding Box": bbox_str,
                                        "Coordinates": f"({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})" if len(bbox) >= 4 else "N/A"
                                    })
                                if table_data:
                                    st.dataframe(table_data, width='stretch')
                            else:
                                # Show clear message when no detections found
                                if results and isinstance(results, list) and len(results) > 0:
                                    st.info(f"**No valid detections found** ({len(results)} invalid result(s))")
                                else:
                                    st.info("**No detections found**")
                            
                            with st.expander("View Full Response"):
                                st.json(result)
                            
                            if idx < len(accepted) - 1:
                                st.markdown("---")
            else:
                st.warning("Crawl finished but no images were sent.")
            st.info(f"Elapsed: {elapsed:.2f}s")
            progress.progress(100)