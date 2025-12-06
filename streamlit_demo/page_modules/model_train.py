"""
Model Training page.
"""
import streamlit as st
import os
import sys
import yaml
import tempfile
import shutil
import traceback

# Try to import tkinter for folder selection dialog
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

# Path setup
_this_dir = os.path.dirname(__file__)
_parent_dir = os.path.abspath(os.path.join(_this_dir, '..'))
_project_root = os.path.abspath(os.path.join(_this_dir, '..', '..'))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Try to import required libraries
try:
    import torch
    from ultralytics import YOLO
    TORCH_AVAILABLE = True
    YOLO_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    YOLO_AVAILABLE = False


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
    except Exception as e:
        st.error(f"Error opening folder dialog: {e}")
        return None


def render():
    """Render the Model Training page."""
    # Add CSS to make buttons full-width
    st.markdown("""
        <style>
        div[data-testid="stButton"] > button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("## Model Training")
    st.markdown("Configure and train a YOLO model using your dataset.")
    
    # Check if required libraries are available
    if not TORCH_AVAILABLE or not YOLO_AVAILABLE:
        st.error("Required libraries (torch, ultralytics) are not available. Please install them using: pip install torch ultralytics")
        return
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available() if TORCH_AVAILABLE else False
    if cuda_available:
        st.success(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("CUDA is not available. Training will use CPU (slower).")
    
    st.markdown("---")
    
    # Section 1: Upload data.yaml file
    st.markdown("### 1. Upload Dataset Configuration")
    
    uploaded_yaml = st.file_uploader(
        "Upload data.yaml file",
        type=["yaml", "yml"],
        help="Upload the YOLO dataset configuration file (data.yaml)"
    )
    
    data_yaml_path = None
    if uploaded_yaml is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode='w') as tmp_file:
            tmp_file.write(uploaded_yaml.getvalue().decode('utf-8'))
            data_yaml_path = tmp_file.name
        
        # Display yaml content preview
        try:
            with open(data_yaml_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
            st.success(f"✅ Successfully loaded: {uploaded_yaml.name}")
            with st.expander("View data.yaml content"):
                st.code(yaml.dump(yaml_content, default_flow_style=False), language='yaml')
        except Exception as e:
            st.error(f"Error reading YAML file: {e}")
    
    st.markdown("---")
    
    # Section 2: Select output directory
    st.markdown("### 2. Select Output Directory")
    
    # Handle folder browsing
    if 'browse_output_dir' in st.session_state and st.session_state.get('browse_output_dir'):
        selected_folder = select_folder_dialog("Select Output Directory for Trained Model")
        if selected_folder:
            st.session_state['model_output_dir'] = selected_folder
        st.session_state['browse_output_dir'] = False
        st.rerun()
    
    output_dir = st.session_state.get('model_output_dir', '')
    
    if output_dir and os.path.exists(output_dir):
        st.success(f"📁 Selected: {output_dir}")
    else:
        st.info("No output directory selected. Click 'Browse files' to select a directory.")
    
    if TKINTER_AVAILABLE:
        if st.button("Browse files", key="browse_output_btn", type="primary"):
            st.session_state['browse_output_dir'] = True
            st.rerun()
    else:
        manual_output_dir = st.text_input(
            "Enter output directory path manually:",
            value=output_dir,
            help="Path to directory where trained model will be saved",
            key="model_output_dir_input_manual"
        )
        if manual_output_dir != output_dir:
            st.session_state['model_output_dir'] = manual_output_dir
            output_dir = manual_output_dir
    
    if output_dir:
        if os.path.exists(output_dir):
            st.caption(f"✓ Valid directory: {os.path.basename(output_dir)}")
        else:
            st.warning(f"⚠ Path not found")
    
    st.markdown("---")
    
    # Section 3: Training Configuration
    st.markdown("### 3. Training Configuration")
    
    with st.expander("Model Selection", expanded=True):
        model_size = st.text_input(
            "YOLO Model Size",
            value="yolov8s.pt",
            help="Enter the YOLO model file (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)"
        )
    
    with st.expander("Hardware Optimization", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=1000,
                value=100,
                step=10,
                help="Number of training epochs"
            )        
            
            batch = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=128,
                value=16,
                step=1,
                help="Batch size for training"
            )

            # Image size input (always visible, but can be disabled)
            # Checkbox below the input
            enable_imgsz = st.checkbox(
                "Configure Image Size",
                value=True,
                help="Enable to configure image size for training",
                key='enable_imgsz'
            )
            
            # Width and Height inputs
            col_width, col_height = st.columns(2)
            
            with col_width:
                imgsz_width = st.number_input(
                    "Image Width",
                    min_value=320,
                    max_value=1280,
                    value=640,
                    step=32,
                    disabled=not enable_imgsz,
                    help="Image width for training (must be multiple of 32)" if enable_imgsz else "No Image size Configured",
                    key='imgsz_width'
                )
            
            with col_height:
                imgsz_height = st.number_input(
                    "Image Height",
                    min_value=320,
                    max_value=1280,
                    value=640,
                    step=32,
                    disabled=not enable_imgsz,
                    help="Image height for training (must be multiple of 32)" if enable_imgsz else "No Image size Configured",
                    key='imgsz_height'
                )
            
            # Set imgsz to None if checkbox is disabled, otherwise create tuple (width, height)
            imgsz = (imgsz_width, imgsz_height) if enable_imgsz else None

        with col2:
            device = st.text_input(
                "Device",
                value="cuda" if cuda_available else "cpu",
                help="Device to use for training (e.g., cuda, cpu, mps)"
            )
            
            workers = st.number_input(
                "Workers",
                min_value=0,
                max_value=16,
                value=6,
                step=1,
                help="Number of worker threads for data loading"
            )
            
            # Cache type input (always visible, but can be disabled)
            # Checkbox below the input
            enable_cache = st.checkbox(
                "Configure Cache Type",
                value=True,
                help="Enable to configure cache type for training",
                key='enable_cache'
            )
            
            cache_type_input = st.text_input(
                "Cache Type",
                value="disk",
                disabled=not enable_cache,
                help="Cache type for faster data loading (e.g., disk, ram)" if enable_cache else "No cache configured"
            )
            
            # Set cache_type to False if checkbox is disabled, otherwise use the input value
            cache_type = cache_type_input if enable_cache else False
    
    with st.expander("Training Parameters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            patience = st.number_input(
                "Patience",
                min_value=0,
                max_value=100,
                value=25,
                step=5,
                help="Early stopping patience (0 = disabled)"
            )
            
            cos_lr = st.checkbox(
                "Cosine LR Scheduler",
                value=True,
                help="Use cosine learning rate scheduler"
            )
        
        with col2:
            save_period = st.number_input(
                "Save Period",
                min_value=0,
                max_value=100,
                value=25,
                step=5,
                help="Save checkpoint every N epochs (0 = disabled)"
            )
    
    with st.expander("Output Configuration", expanded=True):
        model_name = st.text_input(
            "Model Name",
            value="yolov8s-object",
            help="Name for the trained model (will be saved in {output_dir}/train/{model_name})"
        )
    
    st.markdown("---")
    
    # Section 4: Start Training
    st.markdown("### 4. Start Training")
    
    if st.button("🚀 Start Training", type="primary"):
        # Validate inputs
        if not data_yaml_path:
            st.error("Please upload a data.yaml file.")
        elif not output_dir or not os.path.exists(output_dir):
            st.error("Please select a valid output directory.")
        else:
            try:
                # Clear CUDA cache
                if TORCH_AVAILABLE:
                    torch.cuda.empty_cache()
                
                # Load model
                with st.spinner(f"Loading model {model_size}..."):
                    model = YOLO(model_size)
                
                # Prepare training arguments
                train_args = {
                    "data": data_yaml_path,
                    "epochs": epochs,
                    "batch": batch,
                    "device": device,
                    "workers": workers,
                    "patience": patience if patience > 0 else None,
                    "name": model_name,
                    "project": output_dir,  # Use selected output directory
                    "save_period": save_period if save_period > 0 else None,
                }
                
                # Add image size only if configured
                if imgsz is not None:
                    train_args["imgsz"] = imgsz
                
                # Add optional parameters
                if cache_type is not False:
                    if isinstance(cache_type, str) and cache_type.strip():
                        # Convert string to appropriate type
                        cache_value = cache_type.strip().lower()
                        if cache_value == "ram":
                            train_args["cache"] = "ram"
                        elif cache_value == "disk":
                            train_args["cache"] = "disk"
                        else:
                            train_args["cache"] = cache_type.strip()
                    elif cache_type:
                        train_args["cache"] = cache_type
                
                if cos_lr:
                    train_args["cos_lr"] = True
                
                # Display training configuration
                st.markdown("#### Training Configuration Summary")
                config_summary = {
                    "Model": model_size,
                    "Data YAML": uploaded_yaml.name,
                    "Epochs": epochs,
                    "Image Size": f"{imgsz}" if imgsz is not None else "Not configured",
                    "Batch Size": batch,
                    "Device": device,
                    "Workers": workers,
                    "Cache": cache_type if (cache_type and cache_type is not False) else "None",
                    "Patience": patience if patience > 0 else "Disabled",
                    "Cosine LR": "Enabled" if cos_lr else "Disabled",
                    "Save Period": save_period if save_period > 0 else "Disabled",
                    "Model Name": model_name,
                    "Output Directory": output_dir,
                }
                st.json(config_summary)
                
                # Start training
                st.markdown("#### Training Progress")
                st.info("Training started. This may take a while. Progress will be shown below.")
                
                # Create a placeholder for training output
                training_output = st.empty()
                
                # Run training
                with st.spinner("Training in progress..."):
                    results = model.train(**train_args)
                
                # Training completed
                st.success("✅ Training completed successfully!")
                
                # Display results summary
                if results:
                    st.markdown("#### Training Results")
                    st.json({
                        "Best Model": results.save_dir if hasattr(results, 'save_dir') else "N/A",
                        "Metrics": str(results) if results else "N/A"
                    })
                
                # Show where model was saved
                # YOLO saves models to {project}/train/{name}
                runs_dir = os.path.join(output_dir, "train", model_name)
                if os.path.exists(runs_dir):
                    st.info(f"📁 Trained model saved to: `{os.path.abspath(runs_dir)}`")
                    
                    # Find the best.pt file
                    best_model_path = os.path.join(runs_dir, "weights", "best.pt")
                    last_model_path = os.path.join(runs_dir, "weights", "last.pt")
                    
                    if os.path.exists(best_model_path):
                        st.success(f"✅ Best model: `{os.path.abspath(best_model_path)}`")
                    if os.path.exists(last_model_path):
                        st.info(f"📄 Last checkpoint: `{os.path.abspath(last_model_path)}`")
                    
                    # List files in the runs directory
                    with st.expander("View Saved Files"):
                        for root, dirs, files in os.walk(runs_dir):
                            level = root.replace(runs_dir, '').count(os.sep)
                            indent = ' ' * 2 * level
                            st.text(f"{indent}{os.path.basename(root)}/")
                            subindent = ' ' * 2 * (level + 1)
                            for file in files[:20]:  # Limit to first 20 files
                                st.text(f"{subindent}{file}")
                            if len(files) > 20:
                                st.text(f"{subindent}... and {len(files) - 20} more files")
                
                # Clean up temporary file
                if data_yaml_path and os.path.exists(data_yaml_path):
                    os.unlink(data_yaml_path)
                
            except Exception as e:
                st.error(f"Error during training: {e}")
                st.code(traceback.format_exc())
                
                # Clean up temporary file on error
                if data_yaml_path and os.path.exists(data_yaml_path):
                    try:
                        os.unlink(data_yaml_path)
                    except:
                        pass

