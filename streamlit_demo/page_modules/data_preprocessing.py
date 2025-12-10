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
    list_preprocess_profiles,
    apply_preprocess,
)
from app.database import get_preprocess_profile

try:
    from app.preprocess import resize_labels
    RESIZE_LABELS_AVAILABLE = True
except ImportError:
    RESIZE_LABELS_AVAILABLE = False

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
    labels_folder=None,
    progress_callback=None
):
    """
    Process all images in a folder through preprocessing steps.
    Optionally process corresponding labels if resize is applied.
    
    Args:
        input_folder: Directory containing input images
        output_folder: Directory to save preprocessed images (this will be the preprocessed_images folder)
        preprocess_profile: Preprocessing profile to apply
        labels_folder: Optional directory containing YOLO format label files (.txt)
        progress_callback: Optional callback function(processed, total, current_file)
    
    Returns:
        tuple: (processed_count, total_images, errors)
    """
    """
    Process all images in a folder through preprocessing steps.
    Optionally process corresponding labels if resize is applied.
    
    Args:
        input_folder: Directory containing input images
        output_folder: Directory to save preprocessed images
        preprocess_profile: Preprocessing profile to apply
        labels_folder: Optional directory containing YOLO format label files (.txt)
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
            
            # Apply preprocessing - handle both old (2 values) and new (3 values) return formats
            try:
                result = apply_preprocess(img_bytes, preprocess_profile)
                if len(result) == 3:
                    processed_img_bytes, _, resize_info = result
                elif len(result) == 2:
                    # Handle old format that returns only 2 values
                    processed_img_bytes, _ = result
                    resize_info = None
                else:
                    raise ValueError(f"Unexpected return value from apply_preprocess: {len(result)} values")
            except ValueError as ve:
                # Re-raise ValueError with more context
                raise ValueError(f"Error in apply_preprocess for {filename}: {ve}")
            except Exception as e:
                # Wrap other exceptions
                raise Exception(f"Error in apply_preprocess for {filename}: {e}")
            
            # Save preprocessed image
            output_path = os.path.join(output_folder, filename)
            with open(output_path, 'wb') as f:
                f.write(processed_img_bytes)
            
            # Process labels if resize was applied and labels folder is provided
            # Check if preprocessing profile has resize operation (even if resize_info is None due to no actual resize)
            has_resize_operation = any(
                step.get('operation') == 'resize' 
                for step in preprocess_profile.get('steps', [])
            )
            
            if has_resize_operation and labels_folder and RESIZE_LABELS_AVAILABLE:
                label_file = os.path.join(labels_folder, f"{base_name}.txt")
                if os.path.exists(label_file):
                    try:
                        # Read original label
                        with open(label_file, 'r', encoding='utf-8') as f:
                            label_content = f.read()
                        
                        # Resize labels (resize_info might be None if no actual resize occurred, but we still process)
                        if resize_info:
                            resized_label_content = resize_labels(label_content, resize_info)
                        else:
                            # No resize occurred (scale factors are 1.0), but resize operation exists
                            # Just copy the label as-is
                            resized_label_content = label_content
                        
                        # Save resized label to output folder (same parent as output_folder)
                        # output_folder is actually save_images_dir (preprocessed_images), so get its parent
                        output_folder_parent = os.path.dirname(output_folder)
                        output_labels_dir = os.path.join(output_folder_parent, "preprocessed_labels")
                        os.makedirs(output_labels_dir, exist_ok=True)
                        output_label_path = os.path.join(output_labels_dir, f"{base_name}.txt")
                        with open(output_label_path, 'w', encoding='utf-8') as f:
                            f.write(resized_label_content)
                    except Exception as e:
                        errors.append((f"{base_name}.txt (label)", str(e)))
                else:
                    # Label file doesn't exist for this image - this is not necessarily an error
                    # but we could log it if needed
                    pass
            
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
                
                # Special handling for resize operation (returns tuple)
                if operation_name == "resize":
                    img, _ = operation_func(img, params)
                else:
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
    # Add CSS to make buttons full-width
    st.markdown("""
        <style>
        div[data-testid="stButton"] > button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.header("Data Preprocessing")
    st.info("Select a preprocessing profile and upload images to apply the preprocessing steps.")
    
    # Preprocessing profile selection
    try:
        preprocess_profiles = list_preprocess_profiles(limit=100)
        if not preprocess_profiles:
            st.warning("No preprocessing profiles found in MongoDB. Please create a preprocessing profile first in the 'Create and Upload Model' section.")
        else:
            # Create options for selectbox, sorted by preprocess_id in ascending order
            profile_options = {}
            # Sort profiles by preprocess_id to ensure ascending order
            sorted_profiles = sorted(preprocess_profiles, key=lambda p: p.get('preprocess_id', ''))
            for profile in sorted_profiles:
                preprocess_id = profile.get('preprocess_id', 'N/A')
                name = profile.get('name', 'Unnamed')
                # Create display name: "Name (ID)" or just "ID" if no name
                display_name = f"{name} ({preprocess_id})" if name != 'Unnamed' else preprocess_id
                profile_options[display_name] = preprocess_id
            
            selected_profile_display = st.selectbox(
                "Select Preprocessing Profile",
                options=list(profile_options.keys()),
                key="preprocess_profile_select"
            )
            selected_preprocess_id = profile_options[selected_profile_display]
            preprocess_profile = get_preprocess_profile(selected_preprocess_id)
            
            if preprocess_profile:
                st.markdown("### Preprocessing Profile Information")
                
                # Display profile metadata in a more prominent way
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Preprocessing ID", preprocess_profile.get('preprocess_id', 'N/A'))
                with col2:
                    profile_name = preprocess_profile.get('name', 'Unnamed')
                    st.metric("Profile Name", profile_name if profile_name != 'Unnamed' else 'N/A')
                with col3:
                    steps = preprocess_profile.get('steps', [])
                    st.metric("Number of Steps", len(steps) if steps else 0)
                
                # Display preprocessing steps in a table format
                if steps:
                    st.markdown("#### Preprocessing Steps")
                    
                    # Prepare data for table
                    steps_data = []
                    for idx, step in enumerate(steps, 1):
                        operation = step.get('operation', 'Unknown')
                        params = step.get('params', {})
                        
                        # Format parameters as a readable string
                        if params:
                            param_strs = []
                            for key, value in params.items():
                                if isinstance(value, (list, tuple)):
                                    value_str = ', '.join(map(str, value))
                                    param_strs.append(f"{key}: [{value_str}]")
                                else:
                                    param_strs.append(f"{key}: {value}")
                            params_display = "; ".join(param_strs)
                        else:
                            params_display = "No parameters"
                        
                        steps_data.append({
                            "Step": idx,
                            "Operation": operation.capitalize(),
                            "Parameters": params_display
                        })
                    
                    # Display as table
                    df_steps = pd.DataFrame(steps_data)
                    st.dataframe(
                        df_steps,
                        width='stretch',
                        hide_index=True,
                        column_config={
                            "Step": st.column_config.NumberColumn(
                                "Step",
                                help="Processing order",
                                width="small"
                            ),
                            "Operation": st.column_config.TextColumn(
                                "Operation",
                                help="Preprocessing operation name",
                                width="medium"
                            ),
                            "Parameters": st.column_config.TextColumn(
                                "Parameters",
                                help="Operation parameters",
                                width="large"
                            )
                        }
                    )
                else:
                    st.warning("This preprocessing profile has no steps defined.")
                
                st.markdown("---")
                
                # Initialize session state for folders
                if 'preprocess_input_folder' not in st.session_state:
                    st.session_state['preprocess_input_folder'] = ""
                if 'preprocess_output_folder' not in st.session_state:
                    st.session_state['preprocess_output_folder'] = ""
                if 'preprocess_labels_folder' not in st.session_state:
                    st.session_state['preprocess_labels_folder'] = ""
                
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
                    
                    if 'browse_labels_folder' in st.session_state and st.session_state.get('browse_labels_folder'):
                        selected_folder = select_folder_dialog("Select Labels Folder")
                        if selected_folder:
                            st.session_state['preprocess_labels_folder'] = selected_folder
                        st.session_state['browse_labels_folder'] = False
                        st.rerun()
                
                # Folder batch processing section
                st.markdown("### Batch Process Folder")
                with st.expander("Folder Configuration", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Input Images Folder**")
                        input_folder = st.session_state.get('preprocess_input_folder', '')
                        
                        if input_folder and os.path.exists(input_folder):
                            st.success(f"Selected: {input_folder}")
                        else:
                            st.info("No folder selected. Click 'Browse files' to select a folder.")
                        
                        if TKINTER_AVAILABLE:
                            if st.button("Browse files", key="browse_input_btn", type="primary"):
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
                                st.caption(f"Valid folder: {os.path.basename(input_folder)}")
                            else:
                                st.warning(f"Path not found")
                    
                    with col2:
                        st.markdown("**Output Directory**")
                        output_folder = st.session_state.get('preprocess_output_folder', '')
                        
                        if output_folder and os.path.exists(output_folder):
                            st.success(f"Selected: {output_folder}")
                        else:
                            st.info("No folder selected. Click 'Browse files' to select output directory.")
                        
                        if TKINTER_AVAILABLE:
                            if st.button("Browse files", key="browse_output_btn", type="primary"):
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
                                st.caption(f"Valid folder: {os.path.basename(output_folder)}")
                            else:
                                st.warning(f"Path not found")
                    
                    with col3:
                        st.markdown("**Labels Folder (Optional)**")
                        labels_folder = st.session_state.get('preprocess_labels_folder', '')
                                                
                        if labels_folder and os.path.exists(labels_folder):
                            st.success(f"Selected: {labels_folder}")
                        else:
                            st.info("No folder selected. Optional - only needed if resize is applied.")
                        
                        if TKINTER_AVAILABLE:
                            if st.button("Browse files", key="browse_labels_btn", type="primary"):
                                st.session_state['browse_labels_folder'] = True
                                st.rerun()
                        else:
                            manual_labels_folder = st.text_input(
                                "Enter labels folder path manually:",
                                value=labels_folder,
                                help="Path to folder containing YOLO format label files (.txt)",
                                key="labels_folder_input_manual"
                            )
                            if manual_labels_folder != labels_folder:
                                st.session_state['preprocess_labels_folder'] = manual_labels_folder
                                labels_folder = manual_labels_folder
                        
                        if labels_folder:
                            if os.path.exists(labels_folder):
                                st.caption(f"Valid folder: {os.path.basename(labels_folder)}")
                            else:
                                st.warning(f"Path not found")
                
                # Batch process button
                if preprocess_profile and st.button("Process Folder", key="process_folder_button", type="primary"):
                    # Get folder paths from session state
                    input_folder = st.session_state.get('preprocess_input_folder', '')
                    output_folder = st.session_state.get('preprocess_output_folder', '')
                    labels_folder = st.session_state.get('preprocess_labels_folder', '')
                    
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
                                labels_folder=labels_folder if labels_folder and os.path.exists(labels_folder) else None,
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
                    if st.button("Apply Preprocessing", key="apply_preprocess_button", type="primary"):
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
                                    width='stretch'
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
                                        width='stretch'
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
                    st.error("Cannot apply preprocessing: Selected preprocessing profile has no steps defined.")
                
    except Exception as e:
        st.error(f"Error loading preprocessing profiles: {e}")
        st.code(traceback.format_exc())

