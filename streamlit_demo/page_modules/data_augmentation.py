"""
Data Augmentation page.
"""
import streamlit as st
import os
import glob
import shutil
import time
import traceback
from PIL import Image
import sys

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
    import cv2
    import albumentations as A
    import numpy as np
    CV2_AVAILABLE = True
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    ALBUMENTATIONS_AVAILABLE = False


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


def clip_yolo_box(box, eps=1e-6):
    """
    Clip a YOLO bbox [xc, yc, w, h] to [0,1] range with a tiny epsilon
    to avoid strict Albumentations checks failing on -1e-7, 1+1e-7, etc.
    """
    xc, yc, w, h = box
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)

    # Push extremely small negatives to 0 and extremely small >1 to 1
    if xc < 0.0:
        xc = 0.0
    if yc < 0.0:
        yc = 0.0
    if w < 0.0:
        w = 0.0
    if h < 0.0:
        h = 0.0

    if xc > 1.0:
        xc = 1.0
    if yc > 1.0:
        yc = 1.0
    if w > 1.0:
        w = 1.0
    if h > 1.0:
        h = 1.0

    return [xc, yc, w, h]


def augment_images(
    images_dir,
    labels_dir=None,
    save_images_dir=None,
    save_labels_dir=None,
    selected_augmentations=None,
    n_aug=3,
    progress_callback=None,
    mode='random'
):
    """
    Apply selected augmentations to images.
    
    Args:
        images_dir: Directory containing input images
        labels_dir: Directory containing YOLO format label files
        save_images_dir: Directory to save augmented images
        save_labels_dir: Directory to save augmented labels
        selected_augmentations: Dict with augmentation settings
        n_aug: Number of augmentations per image
        progress_callback: Optional callback function(processed, total, current_file)
        mode: 'random' (all augmentations applied randomly to each image) or 
              'sequential' (one augmentation type per augmented image)
    """
    if not CV2_AVAILABLE or not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("OpenCV and albumentations are required for augmentation")
    
    os.makedirs(save_images_dir, exist_ok=True)
    if save_labels_dir:
        os.makedirs(save_labels_dir, exist_ok=True)
    
    has_labels_dir = labels_dir and os.path.exists(labels_dir)
    
    # Initialize variables for both modes
    sequential_transforms = []
    transform_names = []
    transform = None
    
    if mode == 'sequential':
        # Sequential mode: build separate transforms for each augmentation type
        # Order: horizontal_flip, random_rotate90, rotate, brightness, exposure
        
        if selected_augmentations.get('horizontal_flip', False):
            p = selected_augmentations.get('horizontal_flip_p', 0.5)
            if has_labels_dir:
                seq_transform = A.Compose(
                    [A.HorizontalFlip(p=1.0)],
                    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], clip=True)
                )
            else:
                seq_transform = A.Compose([A.HorizontalFlip(p=1.0)])
            sequential_transforms.append(seq_transform)
            transform_names.append('horizontal_flip')
        
        if selected_augmentations.get('random_rotate90', False):
            p = selected_augmentations.get('random_rotate90_p', 1.0)
            if has_labels_dir:
                seq_transform = A.Compose(
                    [A.RandomRotate90(p=1.0)],
                    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], clip=True)
                )
            else:
                seq_transform = A.Compose([A.RandomRotate90(p=1.0)])
            sequential_transforms.append(seq_transform)
            transform_names.append('random_rotate90')
        
        if selected_augmentations.get('rotate', False):
            limit = selected_augmentations.get('rotate_limit', 15)
            if has_labels_dir:
                seq_transform = A.Compose(
                    [A.Rotate(limit=limit, p=1.0)],
                    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], clip=True)
                )
            else:
                seq_transform = A.Compose([A.Rotate(limit=limit, p=1.0)])
            sequential_transforms.append(seq_transform)
            transform_names.append('rotate')
        
        if selected_augmentations.get('brightness', False):
            brightness_limit = selected_augmentations.get('brightness_limit', 0.2)
            contrast_limit = selected_augmentations.get('contrast_limit', 0.0)
            if has_labels_dir:
                seq_transform = A.Compose(
                    [A.RandomBrightnessContrast(
                        brightness_limit=brightness_limit,
                        contrast_limit=contrast_limit,
                        p=1.0
                    )],
                    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], clip=True)
                )
            else:
                seq_transform = A.Compose([A.RandomBrightnessContrast(
                    brightness_limit=brightness_limit,
                    contrast_limit=contrast_limit,
                    p=1.0
                )])
            sequential_transforms.append(seq_transform)
            transform_names.append('brightness')
        
        if selected_augmentations.get('exposure', False):
            gamma_limit = selected_augmentations.get('gamma_limit', (90, 110))
            if has_labels_dir:
                seq_transform = A.Compose(
                    [A.RandomGamma(gamma_limit=gamma_limit, p=1.0)],
                    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], clip=True)
                )
            else:
                seq_transform = A.Compose([A.RandomGamma(gamma_limit=gamma_limit, p=1.0)])
            sequential_transforms.append(seq_transform)
            transform_names.append('exposure')
        
        if not sequential_transforms:
            raise ValueError("No augmentations selected")
        
        # Limit n_aug to number of available transforms
        n_aug = min(n_aug, len(sequential_transforms))
        
    else:
        # Random mode: build single pipeline with all augmentations
        transforms = []
        
        if selected_augmentations.get('horizontal_flip', False):
            p = selected_augmentations.get('horizontal_flip_p', 0.5)
            transforms.append(A.HorizontalFlip(p=p))
        
        if selected_augmentations.get('random_rotate90', False):
            p = selected_augmentations.get('random_rotate90_p', 1.0)
            transforms.append(A.RandomRotate90(p=p))
        
        if selected_augmentations.get('rotate', False):
            limit = selected_augmentations.get('rotate_limit', 15)
            p = selected_augmentations.get('rotate_p', 0.7)
            transforms.append(A.Rotate(limit=limit, p=p))
        
        if selected_augmentations.get('brightness', False):
            brightness_limit = selected_augmentations.get('brightness_limit', 0.2)
            contrast_limit = selected_augmentations.get('contrast_limit', 0.0)
            p = selected_augmentations.get('brightness_p', 0.7)
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=p
            ))
        
        if selected_augmentations.get('exposure', False):
            gamma_limit = selected_augmentations.get('gamma_limit', (90, 110))
            p = selected_augmentations.get('exposure_p', 0.7)
            transforms.append(A.RandomGamma(gamma_limit=gamma_limit, p=p))
        
        if not transforms:
            raise ValueError("No augmentations selected")
        
        # Create transform pipeline
        # Include bbox_params only if labels directory is provided
        if has_labels_dir:
            transform = A.Compose(
                transforms,
                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
            )
        else:
            transform = A.Compose(transforms)
    
    # Get image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        image_files.extend(glob.glob(os.path.join(images_dir, ext.upper())))
    
    image_files = sorted(image_files)
    
    if not image_files:
        raise ValueError(f"No images found in {images_dir}")
    
    total_images = len(image_files)
    processed_count = 0
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        
        # Find corresponding label file (only if labels_dir is provided and exists)
        label_path = None
        if labels_dir and os.path.exists(labels_dir):
            for label_ext in ['.txt']:
                potential_label = os.path.join(labels_dir, base_name + label_ext)
                if os.path.exists(potential_label):
                    label_path = potential_label
                    break
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        h, w = image.shape[:2]
        
        # Read labels if available
        bboxes = []
        class_labels = []
        has_labels = False
        
        if label_path and os.path.exists(label_path):
            has_labels = True
            with open(label_path, 'r') as f:
                lines = f.read().splitlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(float(parts[0]))
                    xc, yc, bw, bh = map(float, parts[1:5])
                    bboxes.append([xc, yc, bw, bh])
                    class_labels.append(cls)
        
        # Save original
        shutil.copy(img_path, os.path.join(save_images_dir, filename))
        if has_labels and save_labels_dir:
            os.makedirs(save_labels_dir, exist_ok=True)
            shutil.copy(label_path, os.path.join(save_labels_dir, base_name + '.txt'))
        
        # Apply augmentations
        if mode == 'sequential':
            # Sequential mode: apply one augmentation type per augmented image
            for i in range(n_aug):
                if i >= len(sequential_transforms):
                    break
                
                aug_name = transform_names[i] if i < len(transform_names) else f"aug{i+1}"
                try:
                    current_transform = sequential_transforms[i]
                    
                    if has_labels and bboxes:
                        transformed = current_transform(
                            image=image,
                            bboxes=bboxes,
                            class_labels=class_labels
                        )
                        out_img = transformed['image']
                        out_boxes = transformed['bboxes']
                        out_labels = transformed['class_labels']
                    else:
                        # No labels, just transform image
                        transformed = current_transform(image=image)
                        out_img = transformed['image']
                        out_boxes = []
                        out_labels = []
                    
                    # Clip boxes to [0,1] for sequential mode
                    if has_labels and out_boxes:
                        clipped_boxes = [clip_yolo_box(b) for b in out_boxes]
                        out_boxes = clipped_boxes
                    
                    out_img_name = f"{base_name}_aug{i+1}_{aug_name}{ext}"
                    out_label_name = f"{base_name}_aug{i+1}_{aug_name}.txt"
                    
                    cv2.imwrite(os.path.join(save_images_dir, out_img_name), out_img)
                    
                    if has_labels and out_boxes and save_labels_dir:
                        os.makedirs(save_labels_dir, exist_ok=True)
                        with open(os.path.join(save_labels_dir, out_label_name), 'w') as f:
                            for l, b in zip(out_labels, out_boxes):
                                f.write(f"{l} {' '.join(map(str, b))}\n")
                except Exception as e:
                    print(f"Error augmenting {filename} (aug {i+1}, {aug_name}): {e}")
                    continue
        else:
            # Random mode: apply all augmentations randomly to each augmented image
            for i in range(n_aug):
                try:
                    if has_labels and bboxes:
                        transformed = transform(
                            image=image,
                            bboxes=bboxes,
                            class_labels=class_labels
                        )
                        out_img = transformed['image']
                        out_boxes = transformed['bboxes']
                        out_labels = transformed['class_labels']
                    else:
                        # No labels, just transform image
                        transformed = transform(image=image)
                        out_img = transformed['image']
                        out_boxes = []
                        out_labels = []
                    
                    out_img_name = f"{base_name}_aug{i+1}{ext}"
                    out_label_name = f"{base_name}_aug{i+1}.txt"
                    
                    cv2.imwrite(os.path.join(save_images_dir, out_img_name), out_img)
                    
                    if has_labels and out_boxes and save_labels_dir:
                        os.makedirs(save_labels_dir, exist_ok=True)
                        with open(os.path.join(save_labels_dir, out_label_name), 'w') as f:
                            for l, b in zip(out_labels, out_boxes):
                                f.write(f"{l} {' '.join(map(str, b))}\n")
                except Exception as e:
                    print(f"Error augmenting {filename} (aug {i+1}): {e}")
                    continue
        
        processed_count += 1
        if progress_callback:
            progress_callback(processed_count, total_images, filename)
    
    return processed_count, total_images


def render():
    """Render the Data Augmentation page."""
    # Add CSS to make buttons full-width
    st.markdown("""
        <style>
        div[data-testid="stButton"] > button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.header("Data Augmentation")
    st.info("Apply data augmentation techniques to expand your training dataset.")
    
    # Check for required libraries
    if not CV2_AVAILABLE or not ALBUMENTATIONS_AVAILABLE:
        st.error("Required libraries not available. Please install: `pip install opencv-python albumentations`")
        return
    
    # Initialize session state for folders
    if 'aug_images_folder' not in st.session_state:
        st.session_state['aug_images_folder'] = ""
    if 'aug_labels_folder' not in st.session_state:
        st.session_state['aug_labels_folder'] = ""
    if 'aug_output_folder' not in st.session_state:
        st.session_state['aug_output_folder'] = ""
    
    # Handle folder selection BEFORE creating widgets
    if TKINTER_AVAILABLE:
        if 'browse_images_folder' in st.session_state and st.session_state.get('browse_images_folder'):
            selected_folder = select_folder_dialog("Select Images Folder")
            if selected_folder:
                st.session_state['aug_images_folder'] = selected_folder
            st.session_state['browse_images_folder'] = False
            st.rerun()
        
        if 'browse_labels_folder' in st.session_state and st.session_state.get('browse_labels_folder'):
            selected_folder = select_folder_dialog("Select Labels Folder")
            if selected_folder:
                st.session_state['aug_labels_folder'] = selected_folder
            st.session_state['browse_labels_folder'] = False
            st.rerun()
        
        if 'browse_output_folder' in st.session_state and st.session_state.get('browse_output_folder'):
            selected_folder = select_folder_dialog("Select Output Directory")
            if selected_folder:
                st.session_state['aug_output_folder'] = selected_folder
            st.session_state['browse_output_folder'] = False
            st.rerun()
    
    # Configuration section
    with st.expander("Folder Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Images Folder**")
            images_folder = st.session_state.get('aug_images_folder', '')
            
            if images_folder and os.path.exists(images_folder):
                st.success(f"Selected: {images_folder}")
            else:
                st.info("No folder selected. Click 'Browse files' to select a folder.")
            
            if TKINTER_AVAILABLE:
                if st.button("Browse files", key="browse_images_btn", type="primary"):
                    st.session_state['browse_images_folder'] = True
                    st.rerun()
            else:
                manual_images_folder = st.text_input(
                    "Enter images folder path manually:",
                    value=images_folder,
                    help="Path to folder containing images",
                    key="images_folder_input_manual"
                )
                if manual_images_folder != images_folder:
                    st.session_state['aug_images_folder'] = manual_images_folder
                    images_folder = manual_images_folder
            
            if images_folder:
                if os.path.exists(images_folder):
                    st.caption(f"✓ Valid folder: {os.path.basename(images_folder)}")
                else:
                    st.warning(f"⚠ Path not found")
        
        with col2:
            st.markdown("**Labels Folder (Optional)**")
            labels_folder = st.session_state.get('aug_labels_folder', '')
            
            if labels_folder and os.path.exists(labels_folder):
                st.success(f"Selected: {labels_folder}")
            else:
                st.info("No folder selected. Click 'Browse files' to select a folder (optional).")
            
            if TKINTER_AVAILABLE:
                if st.button("Browse files", key="browse_labels_btn", type="primary"):
                    st.session_state['browse_labels_folder'] = True
                    st.rerun()
            else:
                manual_labels_folder = st.text_input(
                    "Enter labels folder path manually:",
                    value=labels_folder,
                    help="Path to folder containing YOLO format label files (.txt). Leave empty if no labels.",
                    key="labels_folder_input_manual"
                )
                if manual_labels_folder != labels_folder:
                    st.session_state['aug_labels_folder'] = manual_labels_folder
                    labels_folder = manual_labels_folder
            
            if labels_folder:
                if os.path.exists(labels_folder):
                    st.caption(f"✓ Valid folder: {os.path.basename(labels_folder)}")
                else:
                    st.warning(f"⚠ Path not found")
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**Output Directory**")
            output_folder = st.session_state.get('aug_output_folder', '')
            
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
                    help="Path to directory where aug_images folder will be created",
                    key="output_folder_input_manual"
                )
                if manual_output_folder != output_folder:
                    st.session_state['aug_output_folder'] = manual_output_folder
                    output_folder = manual_output_folder
            
            if output_folder:
                if os.path.exists(output_folder):
                    st.caption(f"✓ Valid folder: {os.path.basename(output_folder)}")
                else:
                    st.warning(f"⚠ Path not found")
        
        with col4:
            st.markdown("**Augmentations per Image**")
            n_aug = st.number_input(
                "Number of augmentations per image:",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                help="How many augmented versions to create for each original image",
                key="n_aug_input"
            )
    
    # Augmentation mode selection
    st.markdown("### Augmentation Mode")
    augmentation_mode = st.radio(
        "Select augmentation mode:",
        options=['random', 'sequential'],
        format_func=lambda x: {
            'random': 'Random Augmentation (all augmentations applied randomly to each image)',
            'sequential': 'Sequential Augmentation (one augmentation type per augmented image)'
        }[x],
        index=0,
        help="Random: All selected augmentations are applied randomly to each augmented image.\n"
             "\nSequential: Each augmented image gets only one specific augmentation type (aug1 = first selected, aug2 = second selected, etc.)",
        key="augmentation_mode"
    )
    
    # Augmentation selection
    st.markdown("### Select Augmentations")
    
    selected_augmentations = {}
    
    with st.expander("Transformation Augmentations", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            selected_augmentations['horizontal_flip'] = st.checkbox(
                "Horizontal Flip",
                value=False,
                help="Flip image horizontally"
            )
            if selected_augmentations['horizontal_flip']:
                selected_augmentations['horizontal_flip_p'] = st.slider(
                    "Horizontal Flip Probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    key="horizontal_flip_p"
                )
            
            selected_augmentations['random_rotate90'] = st.checkbox(
                "Random Rotate 90°",
                value=False,
                help="Rotate image by 90° (clockwise or counterclockwise)"
            )
            if selected_augmentations['random_rotate90']:
                selected_augmentations['random_rotate90_p'] = st.slider(
                    "Random Rotate 90° Probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0,
                    step=0.1,
                    key="random_rotate90_p"
                )
        
        with col2:
            selected_augmentations['rotate'] = st.checkbox(
                "Rotate",
                value=False,
                help="Random rotation within specified degree limit"
            )
            if selected_augmentations['rotate']:
                selected_augmentations['rotate_limit'] = st.number_input(
                    "Rotation Degree Limit",
                    min_value=1,
                    max_value=180,
                    value=15,
                    step=1,
                    help="Maximum rotation angle in degrees (e.g., 15 means -15° to +15°)",
                    key="rotate_limit"
                )
                selected_augmentations['rotate_p'] = st.slider(
                    "Rotation Probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    key="rotate_p"
                )
    
    with st.expander("Color & Brightness Augmentations", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            selected_augmentations['brightness'] = st.checkbox(
                "Brightness & Contrast",
                value=False,
                help="Adjust brightness and contrast"
            )
            if selected_augmentations['brightness']:
                selected_augmentations['brightness_limit'] = st.slider(
                    "Brightness Limit",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                    step=0.05,
                    help="Brightness adjustment range (e.g., 0.2 means -20% to +20%)",
                    key="brightness_limit"
                )
                selected_augmentations['contrast_limit'] = st.slider(
                    "Contrast Limit",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    help="Contrast adjustment range",
                    key="contrast_limit"
                )
                selected_augmentations['brightness_p'] = st.slider(
                    "Brightness Probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    key="brightness_p"
                )
        
        with col2:
            selected_augmentations['exposure'] = st.checkbox(
                "Random Gamma (Exposure-like)",
                value=False,
                help="Adjust image gamma for exposure-like changes"
            )
            if selected_augmentations['exposure']:
                gamma_min = st.number_input(
                    "Gamma Min",
                    min_value=50,
                    max_value=150,
                    value=90,
                    step=5,
                    help="Minimum gamma value (lower = darker)",
                    key="gamma_min"
                )
                gamma_max = st.number_input(
                    "Gamma Max",
                    min_value=50,
                    max_value=150,
                    value=110,
                    step=5,
                    help="Maximum gamma value (higher = brighter)",
                    key="gamma_max"
                )
                if gamma_min >= gamma_max:
                    st.warning("Gamma Min should be less than Gamma Max")
                selected_augmentations['gamma_limit'] = (gamma_min, gamma_max)
                selected_augmentations['exposure_p'] = st.slider(
                    "Gamma Probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    key="exposure_p"
                )
    
    # Apply augmentation button
    if st.button("Apply Augmentation", type="primary"):
        # Get folder paths from session state
        images_folder = st.session_state.get('aug_images_folder', '')
        labels_folder = st.session_state.get('aug_labels_folder', '')
        output_folder = st.session_state.get('aug_output_folder', '')
        
        # Validate inputs
        if not images_folder:
            st.error("Please select an images folder.")
            return
        
        if not os.path.exists(images_folder):
            st.error(f"Images folder not found: {images_folder}")
            return
        
        if not output_folder:
            st.error("Please select an output directory.")
            return
        
        if not os.path.exists(output_folder):
            st.error(f"Output directory not found: {output_folder}")
            return
        
        # Check if any augmentation is selected
        has_selection = any([
            selected_augmentations.get('horizontal_flip', False),
            selected_augmentations.get('random_rotate90', False),
            selected_augmentations.get('rotate', False),
            selected_augmentations.get('brightness', False),
            selected_augmentations.get('exposure', False),
        ])
        
        if not has_selection:
            st.error("Please select at least one augmentation.")
            return
        
        # Set up output paths
        save_images_dir = os.path.join(output_folder, "aug_images")
        save_labels_dir = os.path.join(output_folder, "aug_labels") if labels_folder and os.path.exists(labels_folder) else None
        
        # Status container
        status_text = st.empty()
        status_text.info("Starting augmentation process...")
        
        try:
            # Progress tracking
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            def progress_callback(processed, total, current_file):
                progress = processed / total
                progress_bar.progress(progress)
                progress_text.text(f"Processing: {current_file} ({processed}/{total})")
            
            # Apply augmentations
            processed_count, total_images = augment_images(
                images_dir=images_folder,
                labels_dir=labels_folder if labels_folder and os.path.exists(labels_folder) else None,
                save_images_dir=save_images_dir,
                save_labels_dir=save_labels_dir,
                selected_augmentations=selected_augmentations,
                n_aug=n_aug,
                progress_callback=progress_callback,
                mode=augmentation_mode
            )
            
            # Clear progress
            progress_bar.empty()
            progress_text.empty()
            
            # Show completion
            status_text.success(f"Augmentation Complete! Processed {processed_count}/{total_images} images. Augmented images saved to: {save_images_dir}")
            
            # Display summary
            st.markdown("### Augmentation Summary")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            with summary_col1:
                st.metric("Images Processed", processed_count)
            with summary_col2:
                st.metric("Augmentations per Image", n_aug)
            with summary_col3:
                st.metric("Total Augmented Images", processed_count * n_aug)
            
            st.info(f"**Output Location:**\n- Images: `{save_images_dir}`\n" + 
                   (f"- Labels: `{save_labels_dir}`" if save_labels_dir else "- Labels: Not saved (no labels folder provided)"))
            
        except Exception as e:
            status_text.error(f"Error during augmentation: {e}")
            st.code(traceback.format_exc())
    
    if not TKINTER_AVAILABLE:
        st.info("**Tip:** To use folder browser, install tkinter: `pip install tk` (Linux) or it's included with Python on Windows/Mac.")
