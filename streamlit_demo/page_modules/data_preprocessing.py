"""
Data Preprocessing page.
"""
import streamlit as st
import io
import traceback
import pandas as pd
import time
from PIL import Image
import glob
import shutil

import sys
import os
_this_dir = os.path.dirname(__file__)
_parent_dir = os.path.abspath(os.path.join(_this_dir, '..'))
_project_root = os.path.abspath(os.path.join(_this_dir, '..', '..'))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Try to import tkinter for folder selection dialog
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

from utils import (
    list_models,
    get_model_by_id,
    get_preprocess_for_model,
    apply_preprocess,
)

try:
    from app.preprocess import OPERATION_REGISTRY, _bytes_to_cv2_image, _cv2_image_to_bytes
    PREPROCESS_AVAILABLE = True
except ImportError:
    PREPROCESS_AVAILABLE = False


def select_folder_dialog(title="Select Folder"):
    """Open a folder selection dialog using tkinter."""
    if not TKINTER_AVAILABLE:
        return None
    
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder_path = filedialog.askdirectory(title=title)
        root.destroy()
        return folder_path if folder_path else None
    except Exception:
        return None


def process_folder_images(
    input_folder,
    output_folder,
    preprocess_profile,
    progress_callback=None
):
    """
    Process all images in a folder through preprocessing steps.
    
    Args:
        input_folder: Directory containing input images
        output_folder: Directory to save preprocessed images
        preprocess_profile: Preprocessing profile to apply
        progress_callback: Optional callback function(processed, total, current_file)
    
    Returns:
        tuple: (processed_count, total_images, errors)
    """
    if not PREPROCESS_AVAILABLE or not preprocess_profile:
        raise ImportError("Preprocessing functionality not available")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Get image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    image_files = sorted(image_files)
    
    if not image_files:
        raise ValueError(f"No images found in {input_folder}")
    
    total_images = len(image_files)
    processed_count = 0
    errors = []
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        
        try:
            # Read image file
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            
            # Apply preprocessing
            processed_img_bytes, _ = apply_preprocess(img_bytes, preprocess_profile)
            
            # Save preprocessed image
            output_path = os.path.join(output_folder, filename)
            with open(output_path, 'wb') as f:
                f.write(processed_img_bytes)
            
            processed_count += 1
            if progress_callback:
                progress_callback(processed_count, total_images, filename)
        except Exception as e:
            errors.append((filename, str(e)))
            continue
    
    return processed_count, total_images, errors


def apply_preprocess_step_by_step(img_bytes, preprocess_profile):
    """
    Apply preprocessing steps one by one and return intermediate results.
    
    Returns:
        list of tuples: [(step_info, intermediate_img_bytes), ...]
        where step_info contains operation name and params
    """
    if not PREPROCESS_AVAILABLE or not preprocess_profile:
        return []
    
    steps = preprocess_profile.get("steps", [])
    if not steps:
        return []
    
    try:
        # Convert bytes to OpenCV image
        img = _bytes_to_cv2_image(img_bytes)
        results = []
        
        # Apply each step in sequence and capture intermediate results
        for step in steps:
            operation_name = step.get("operation")
            params = step.get("params", {})
            
            if operation_name not in OPERATION_REGISTRY:
                continue
            
            try:
                operation_func = OPERATION_REGISTRY[operation_name]
                img = operation_func(img, params)
                
                # Convert intermediate result to bytes for display
                intermediate_bytes = _cv2_image_to_bytes(img, format='PNG')
                results.append({
                    "operation": operation_name,
                    "params": params,
                    "image_bytes": intermediate_bytes
                })
            except Exception as e:
                # Continue with next step on error
                continue
        
        return results
        
    except Exception as e:
        return []


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
                        st.warning("This model has a preprocessing profile but no steps defined.")
                else:
                    st.warning("This model does not have a preprocessing profile configured.")
                
                st.markdown("---")
                
                # Initialize session state for folders
                if 'preprocess_input_folder' not in st.session_state:
                    st.session_state['preprocess_input_folder'] = ""
                if 'preprocess_output_folder' not in st.session_state:
                    st.session_state['preprocess_output_folder'] = ""
                
                # Handle folder selection BEFORE creating widgets
                if TKINTER_AVAILABLE:
                    if 'browse_input_folder' in st.session_state and st.session_state.get('browse_input_folder'):
                        selected_folder = select_folder_dialog("Select Images Folder")
                        if selected_folder:
                            st.session_state['preprocess_input_folder'] = selected_folder
                        st.session_state['browse_input_folder'] = False
                        st.rerun()
                    
                    if 'browse_output_folder' in st.session_state and st.session_state.get('browse_output_folder'):
                        selected_folder = select_folder_dialog("Select Output Directory")
                        if selected_folder:
                            st.session_state['preprocess_output_folder'] = selected_folder
                        st.session_state['browse_output_folder'] = False
                        st.rerun()
                
                # Folder batch processing section
                st.markdown("### Batch Process Folder")
                with st.expander("Folder Configuration", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Input Images Folder**")
                        input_folder = st.session_state.get('preprocess_input_folder', '')
                        
                        if input_folder and os.path.exists(input_folder):
                            st.success(f"Selected: {input_folder}")
                        else:
                            st.info("No folder selected. Click 'Browse files' to select a folder.")
                        
                        if TKINTER_AVAILABLE:
                            if st.button("Browse files", key="browse_input_btn", use_container_width=True, type="primary"):
                                st.session_state['browse_input_folder'] = True
                                st.rerun()
                        else:
                            manual_input_folder = st.text_input(
                                "Enter images folder path manually:",
                                value=input_folder,
                                help="Path to folder containing images",
                                key="input_folder_input_manual"
                            )
                            if manual_input_folder != input_folder:
                                st.session_state['preprocess_input_folder'] = manual_input_folder
                                input_folder = manual_input_folder
                        
                        if input_folder:
                            if os.path.exists(input_folder):
                                st.caption(f"✓ Valid folder: {os.path.basename(input_folder)}")
                            else:
                                st.warning(f"⚠ Path not found")
                    
                    with col2:
                        st.markdown("**Output Directory**")
                        output_folder = st.session_state.get('preprocess_output_folder', '')
                        
                        if output_folder and os.path.exists(output_folder):
                            st.success(f"Selected: {output_folder}")
                        else:
                            st.info("No folder selected. Click 'Browse files' to select output directory.")
                        
                        if TKINTER_AVAILABLE:
                            if st.button("Browse files", key="browse_output_btn", use_container_width=True, type="primary"):
                                st.session_state['browse_output_folder'] = True
                                st.rerun()
                        else:
                            manual_output_folder = st.text_input(
                                "Enter output directory path manually:",
                                value=output_folder,
                                help="Path to directory where preprocessed_images folder will be created",
                                key="output_folder_input_manual"
                            )
                            if manual_output_folder != output_folder:
                                st.session_state['preprocess_output_folder'] = manual_output_folder
                                output_folder = manual_output_folder
                        
                        if output_folder:
                            if os.path.exists(output_folder):
                                st.caption(f"✓ Valid folder: {os.path.basename(output_folder)}")
                            else:
                                st.warning(f"⚠ Path not found")
                
                # Batch process button
                if preprocess_profile and st.button("Process Folder", key="process_folder_button", type="primary", use_container_width=True):
                    # Get folder paths from session state
                    input_folder = st.session_state.get('preprocess_input_folder', '')
                    output_folder = st.session_state.get('preprocess_output_folder', '')
                    
                    # Validate inputs
                    if not input_folder:
                        st.error("Please select an input images folder.")
                    elif not os.path.exists(input_folder):
                        st.error(f"Input folder not found: {input_folder}")
                    elif not output_folder:
                        st.error("Please select an output directory.")
                    elif not os.path.exists(output_folder):
                        st.error(f"Output directory not found: {output_folder}")
                    else:
                        # Set up output path
                        save_images_dir = os.path.join(output_folder, "preprocessed_images")
                        
                        # Status container
                        status_text = st.empty()
                        status_text.info("Starting batch preprocessing...")
                        
                        try:
                            # Progress tracking
                            progress_bar = st.progress(0)
                            progress_text = st.empty()
                            
                            def progress_callback(processed, total, current_file):
                                progress = processed / total
                                progress_bar.progress(progress)
                                progress_text.text(f"Processing: {current_file} ({processed}/{total})")
                            
                            # Process folder
                            processed_count, total_images, errors = process_folder_images(
                                input_folder=input_folder,
                                output_folder=save_images_dir,
                                preprocess_profile=preprocess_profile,
                                progress_callback=progress_callback
                            )
                            
                            # Clear progress
                            progress_bar.empty()
                            progress_text.empty()
                            
                            # Show completion
                            status_text.success(f"Batch Preprocessing Complete! Processed {processed_count}/{total_images} images. Preprocessed images saved to: {save_images_dir}")
                            
                            # Display summary
                            st.markdown("### Batch Processing Summary")
                            summary_col1, summary_col2 = st.columns(2)
                            with summary_col1:
                                st.metric("Images Processed", processed_count)
                            with summary_col2:
                                st.metric("Total Images", total_images)
                            
                            if errors:
                                st.warning(f"{len(errors)} images failed to process:")
                                with st.expander("View Errors"):
                                    for filename, error_msg in errors:
                                        st.text(f"{filename}: {error_msg}")
                            
                            st.info(f"**Output Location:** `{save_images_dir}`")
                            
                        except Exception as e:
                            status_text.error(f"Error during batch preprocessing: {e}")
                            st.code(traceback.format_exc())
                
                st.markdown("---")
                
                # Single image upload section
                st.markdown("### Upload Single Image for Preprocessing")
                uploaded_image = st.file_uploader(
                    "Upload Image",
                    type=["jpg", "jpeg", "png", "bmp"],
                    key="preprocess_image_upload"
                )
                
                if uploaded_image and preprocess_profile:
                    if st.button("Apply Preprocessing", key="apply_preprocess_button"):
                        try:
                            # Read original image bytes
                            original_img_bytes = uploaded_image.read()
                            
                            # Get preprocessing steps
                            steps = preprocess_profile.get('steps', [])
                            
                            if not steps:
                                st.warning("No preprocessing steps defined in the profile.")
                                return
                            
                            # Display header
                            st.markdown("### Preprocessing Steps Output")
                            
                            # Load original image
                            original_img = Image.open(io.BytesIO(original_img_bytes))
                            
                            # Apply preprocessing step by step
                            step_results = apply_preprocess_step_by_step(original_img_bytes, preprocess_profile)
                            
                            if not step_results:
                                st.warning("No preprocessing steps were successfully applied.")
                                return
                            
                            # Status text for progress messages (keep visible throughout)
                            status_text = st.empty()
                            
                            # Display original image first in grid (3 columns)
                            cols = st.columns(3)
                            with cols[0]:
                                st.image(
                                    original_img,
                                    caption=f"Original ({original_img.size[0]} × {original_img.size[1]})",
                                    use_container_width=True
                                )
                            
                            # Store all step images for display after processing
                            step_images = []
                            
                            # Process each step sequentially with 1s delays
                            for idx, step_result in enumerate(step_results, 1):
                                operation_name = step_result.get('operation', 'Unknown')
                                step_img_bytes = step_result.get('image_bytes')
                                
                                # Show "Applying..." message (keep container visible)
                                status_text.info(f"Applying {operation_name.capitalize()}... (Step {idx}/{len(step_results)})")
                                
                                # Wait 1 second
                                time.sleep(1.0)
                                
                                # Store image for later display
                                if step_img_bytes:
                                    step_img = Image.open(io.BytesIO(step_img_bytes))
                                    step_images.append((operation_name, step_img))
                            
                            # After all processing is complete, change to green and show completion message
                            status_text.success(f"Preprocessing Complete! All {len(step_results)} steps applied successfully.")
                            
                            # Wait a moment to show completion message
                            time.sleep(0.5)
                            
                            # Display all processed images in grid
                            for idx, (operation_name, step_img) in enumerate(step_images, 1):
                                # Calculate grid position (3 columns)
                                col_idx = idx % 3
                                
                                # Create new row if needed (when col_idx == 0, meaning we've filled 3 columns)
                                if col_idx == 0:
                                    cols = st.columns(3)
                                
                                # Display in appropriate column
                                with cols[col_idx]:
                                    step_width, step_height = step_img.size
                                    st.image(
                                        step_img,
                                        caption=f"{operation_name.capitalize()} ({step_width} × {step_height})",
                                        use_container_width=True
                                    )
                            
                            # Display final summary
                            st.markdown("---")
                            st.markdown("### Preprocessing Summary")
                            
                            # Final processed image
                            final_img = Image.open(io.BytesIO(step_results[-1]['image_bytes']))
                            final_width, final_height = final_img.size
                            
                            st.markdown("#### Final Processed Image")
                            st.image(
                                final_img,
                                caption=f"Final Result ({final_width} × {final_height})",
                                width=400
                            )
                            
                            # Image size comparison
                            st.markdown("### Image Size Comparison")
                            size_col1, size_col2 = st.columns(2)
                            with size_col1:
                                st.metric("Original Size", f"{original_img.size[0]} × {original_img.size[1]}")
                            with size_col2:
                                st.metric("Final Processed Size", f"{final_width} × {final_height}")
                            
                            # Steps summary table
                            st.markdown("### Applied Preprocessing Steps")
                            steps_data = []
                            for idx, step_result in enumerate(step_results, 1):
                                operation = step_result.get('operation', 'Unknown')
                                params = step_result.get('params', {})
                                steps_data.append({
                                    "Step": idx,
                                    "Operation": operation.capitalize(),
                                    "Parameters": str(params) if params else "None"
                                })
                            
                            df_steps = pd.DataFrame(steps_data)
                            st.dataframe(df_steps, width='stretch', hide_index=True)
                            
                        except Exception as e:
                            st.error(f"Error applying preprocessing: {e}")
                            st.code(traceback.format_exc())
                
                elif uploaded_image and not preprocess_profile:
                    st.error("Cannot apply preprocessing: Selected model does not have a preprocessing profile.")
                
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.code(traceback.format_exc())

