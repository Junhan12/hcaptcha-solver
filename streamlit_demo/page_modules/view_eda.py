"""
View EDA (Exploratory Data Analysis) page.
Displays graphs generated from eda.ipynb notebook.
"""
import streamlit as st
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import os
import glob
from collections import Counter
import sys
from PIL import Image

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
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


def variance_of_laplacian(image):
    """Compute the Laplacian of the image and return the variance."""
    return cv2.Laplacian(image, cv2.CV_64F).var()


def analyze_color_distribution(dataset_folder, file_extension="*.jpg"):
    """Analyze color distribution (Hue and Saturation) across dataset."""
    image_paths = glob.glob(os.path.join(dataset_folder, file_extension))
    
    if not image_paths:
        st.warning(f"No images found in {dataset_folder} with extension {file_extension}")
        return None

    hue_hist_acc = np.zeros((180, 1))
    sat_hist_acc = np.zeros((256, 1))

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Processing {len(image_paths)} images for Color Analysis...")

    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        sat_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])

        hue_hist_acc += hue_hist
        sat_hist_acc += sat_hist

        if (idx + 1) % 50 == 0:
            progress_bar.progress((idx + 1) / len(image_paths))

    progress_bar.progress(1.0)
    status_text.empty()

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Hue Plot
    ax1.plot(hue_hist_acc, color='orange')
    ax1.set_title('Hue Distribution (Color Spectrum)')
    ax1.set_xlabel('Pixel Value (0-179)')
    ax1.set_ylabel('Frequency')
    ax1.fill_between(range(180), hue_hist_acc.flatten(), color='orange', alpha=0.3)
    ax1.text(0, 0, "Red", color='red')
    ax1.text(60, 0, "Green", color='green')
    ax1.text(120, 0, "Blue", color='blue')

    # Saturation Plot
    ax2.plot(sat_hist_acc, color='blue')
    ax2.set_title('Saturation Distribution (Vibrancy)')
    ax2.set_xlabel('Pixel Value (0-255)')
    ax2.set_ylabel('Frequency')
    ax2.fill_between(range(256), sat_hist_acc.flatten(), color='blue', alpha=0.3)
    ax2.text(10, 0, "Gray/Faded", color='gray')
    ax2.text(240, 0, "Vibrant", color='blue')

    plt.tight_layout()
    return fig


def analyze_brightness_contrast(dataset_folder, file_extension="*.jpg"):
    """Analyze brightness and contrast distribution."""
    image_paths = glob.glob(os.path.join(dataset_folder, file_extension))
    
    if not image_paths:
        st.warning(f"No images found in {dataset_folder} with extension {file_extension}")
        return None

    brightness_values = []
    contrast_values = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Processing {len(image_paths)} images for Brightness/Contrast...")

    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_contrast = np.std(gray)

        brightness_values.append(mean_brightness)
        contrast_values.append(std_contrast)

        if (idx + 1) % 50 == 0:
            progress_bar.progress((idx + 1) / len(image_paths))

    progress_bar.progress(1.0)
    status_text.empty()

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Brightness
    ax1.hist(brightness_values, bins=30, color='green', alpha=0.7)
    ax1.set_title('Brightness Distribution')
    ax1.set_xlabel('Average Pixel Intensity (0=Black, 255=White)')
    ax1.set_ylabel('Count of Images')

    # Contrast
    ax2.hist(contrast_values, bins=30, color='purple', alpha=0.7)
    ax2.set_title('Contrast Distribution')
    ax2.set_xlabel('Standard Deviation (Low=Gray, High=High Contrast)')
    ax2.set_ylabel('Count of Images')

    plt.tight_layout()

    return fig


def analyze_blur(dataset_folder, file_extension="*.jpg"):
    """Analyze blur distribution using Laplacian variance."""
    image_paths = glob.glob(os.path.join(dataset_folder, file_extension))
    
    if not image_paths:
        st.warning(f"No images found in {dataset_folder} with extension {file_extension}")
        return None

    blur_scores = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Processing {len(image_paths)} images for Blur Detection...")

    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = variance_of_laplacian(gray)
        blur_scores.append(score)

        if (idx + 1) % 50 == 0:
            progress_bar.progress((idx + 1) / len(image_paths))

    progress_bar.progress(1.0)
    status_text.empty()

    # Plotting
    fig = plt.figure(figsize=(10, 6))
    plt.hist(blur_scores, bins=30, color='red', alpha=0.7)
    plt.title('Blur Score Distribution (Laplacian Variance)')
    plt.xlabel('Blur Score (Lower = Blurrier)')
    plt.ylabel('Count of Images')

    return fig


def plot_class_distribution(label_folder, class_names):
    """Plot class distribution from label files."""
    label_files = glob.glob(os.path.join(label_folder, "*.txt"))
    
    if not label_files:
        st.warning(f"No label files found in {label_folder}")
        return None

    class_counts = Counter()

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Scanning {len(label_files)} label files...")

    for idx, file in enumerate(label_files):
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
        except Exception:
            continue

        if (idx + 1) % 50 == 0:
            progress_bar.progress((idx + 1) / len(label_files))

    progress_bar.progress(1.0)
    status_text.empty()

    # Prepare data for plotting
    counts = [class_counts.get(i, 0) for i in range(len(class_names))]
    
    # Plotting
    fig = plt.figure(figsize=(9, 5))
    bars = plt.bar(class_names, counts, color='skyblue', edgecolor='black')
    
    # Add numbers on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), 
                ha='center', va='bottom')

    plt.title('Class Distribution (Balance Check)')
    plt.xlabel('Classes')
    plt.ylabel('Number of Instances')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')

    return fig


def plot_aspect_ratio_separate(label_folder, class_names):
    """Plot bounding box aspect ratio distribution with separate subplot per class."""
    import math
    
    label_files = glob.glob(os.path.join(label_folder, "*.txt"))
    
    if not label_files:
        st.warning(f"No label files found in {label_folder}")
        return None

    class_data = {name: [] for name in class_names}

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Scanning {len(label_files)} label files...")

    for idx, file in enumerate(label_files):
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        w = float(parts[3])
                        h = float(parts[4])
                        
                        if 0 <= class_id < len(class_names):
                            class_data[class_names[class_id]].append((w, h))
        except Exception:
            continue

        if (idx + 1) % 50 == 0:
            progress_bar.progress((idx + 1) / len(label_files))

    progress_bar.progress(1.0)
    status_text.empty()

    # Determine grid size
    num_classes = len(class_names)
    cols = 4
    rows = math.ceil(num_classes / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    axes = axes.flatten()

    # Define colors for distinction
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

    for i, class_name in enumerate(class_names):
        ax = axes[i]
        points = class_data[class_name]
        
        # Plot the dots
        if points:
            ws, hs = zip(*points)
            ax.scatter(ws, hs, alpha=0.5, color=colors[i % len(colors)], s=15)
        
        # Add the 1:1 Square Line (Reference)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label="1:1 Ratio")
        
        # Formatting
        ax.set_title(f"{class_name} ({len(points)})", fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Highlight the "Square vs Diamond" Check
        if class_name in ["square", "diamond"]:
            ax.set_facecolor('#f0f8ff')  # Light blue background for attention

    # Hide empty subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Aspect Ratio Distribution by Class', fontsize=16)
    plt.tight_layout()

    # Generate data table
    table_data = []
    for name in class_names:
        points = class_data[name]
        if not points:
            table_data.append({
                'Class Name': name,
                'Count': 0,
                'Mean W': '-',
                'Mean H': '-',
                'Aspect Ratio (W/H)': '-',
                'Consistency (Std)': '-'
            })
            continue
        
        # Convert to numpy array for fast math
        pts_array = np.array(points)
        ws = pts_array[:, 0]
        hs = pts_array[:, 1]
        
        # Calculate Ratios (Width / Height)
        ratios = ws / hs 
        
        count = len(points)
        mean_w = np.mean(ws)
        mean_h = np.mean(hs)
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)  # Low std dev = Very consistent shape
        
        table_data.append({
            'Class Name': name,
            'Count': count,
            'Mean W': f"{mean_w:.3f}",
            'Mean H': f"{mean_h:.3f}",
            'Aspect Ratio (W/H)': f"{mean_ratio:.3f}",
            'Consistency (Std)': f"{std_ratio:.3f}"
        })
    
    # Display table in Streamlit
    if table_data:
        st.markdown("#### Aspect Ratio Statistics")
        st.dataframe(
            table_data,
            width='stretch',
            hide_index=True
        )

    return fig


def plot_objects_per_image(label_folder):
    """Plot distribution of objects per image."""
    label_files = glob.glob(os.path.join(label_folder, "*.txt"))
    
    if not label_files:
        st.warning(f"No label files found in {label_folder}")
        return None

    counts = []
    empty_files = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Processing {len(label_files)} label files...")

    for idx, file in enumerate(label_files):
        try:
            with open(file, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                count = len(lines)
                counts.append(count)
                
                if count == 0:
                    empty_files.append(os.path.basename(file))
        except Exception:
            continue

        if (idx + 1) % 50 == 0:
            progress_bar.progress((idx + 1) / len(label_files))

    progress_bar.progress(1.0)
    status_text.empty()

    if not counts:
        st.warning("No valid label data found.")
        return None

    # Plotting
    fig = plt.figure(figsize=(8, 5))
    plt.hist(counts, bins=range(min(counts), max(counts) + 2), align='left', 
             color='purple', edgecolor='black')
    
    plt.title('Object Density (Objects per Image)')
    plt.xlabel('Count of Objects')
    plt.ylabel('Number of Images')
    plt.xticks(range(min(counts), max(counts) + 1))
    plt.grid(axis='y', alpha=0.5)

    # Display statistics
    avg_objects = sum(counts) / len(counts)
    st.metric("Average objects per image", f"{avg_objects:.2f}")
    
    if empty_files:
        st.warning(f"Found {len(empty_files)} empty label files (Background images).")

    return fig


def plot_image_sizes(dataset_folder, file_extension="*.jpg"):
    """Plot image size distribution."""
    image_paths = glob.glob(os.path.join(dataset_folder, file_extension))
    
    if not image_paths:
        st.warning(f"No images found in {dataset_folder} with extension {file_extension}")
        return None

    widths = []
    heights = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Scanning dimensions of {len(image_paths)} images...")

    for idx, path in enumerate(image_paths):
        try:
            with Image.open(path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception as e:
            st.warning(f"Error reading {path}: {e}")
            continue

        if (idx + 1) % 50 == 0:
            progress_bar.progress((idx + 1) / len(image_paths))

    progress_bar.progress(1.0)
    status_text.empty()

    if not widths:
        st.warning("No valid images found.")
        return None

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(widths, heights, color='blue', alpha=0.5, edgecolors='k', s=50)
    
    plt.title('Image Size Distribution (Width vs Height)')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Console Report
    avg_w = sum(widths) / len(widths)
    avg_h = sum(heights) / len(heights)
    
    # Display statistics
    st.markdown("#### Image Size Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", len(widths))
    with col2:
        st.metric("Max Size", f"{max(widths)}x{max(heights)}")
    with col3:
        st.metric("Min Size", f"{min(widths)}x{min(heights)}")
    with col4:
        st.metric("Average Size", f"{int(avg_w)}x{int(avg_h)}")
    
    # Check for consistency
    unique_sizes = set(zip(widths, heights))
    if len(unique_sizes) == 1:
        st.success("✅ PERFECT CONSISTENCY: All images are exactly the same size.")
    else:
        st.warning(f"⚠️ VARIATION DETECTED: Found {len(unique_sizes)} different image sizes.")

    return fig


def select_folder_dialog(title="Select Folder"):
    """Open a folder selection dialog using tkinter."""
    if not TKINTER_AVAILABLE:
        return None
    
    try:
        # Hide the root window
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Open folder dialog
        folder_path = filedialog.askdirectory(title=title)
        root.destroy()
        
        return folder_path if folder_path else None
    except Exception:
        return None


def render():
    """Render the View EDA page."""
    st.header("View EDA (Exploratory Data Analysis)")
    st.info("This section provides exploratory data analysis of your collected dataset.")
    
    # Add CSS styling for graph width control (80% width, max-width 100%)
    st.markdown(
        """
        <style>
        .stPlotlyChart, .element-container img, [data-testid="stImage"] {
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Configuration section
    with st.expander("Dataset Configuration", expanded=True):
        # Initialize session state for folders (only if not already set)
        if 'dataset_folder' not in st.session_state:
            st.session_state['dataset_folder'] = ""
        if 'label_folder' not in st.session_state:
            st.session_state['label_folder'] = ""
        
        # Handle folder selection BEFORE creating widgets
        if TKINTER_AVAILABLE:
            if 'browse_dataset' in st.session_state and st.session_state.get('browse_dataset'):
                selected_folder = select_folder_dialog("Select Dataset Folder (Images)")
                if selected_folder:
                    st.session_state['dataset_folder'] = selected_folder
                st.session_state['browse_dataset'] = False
                st.rerun()
            
            if 'browse_label' in st.session_state and st.session_state.get('browse_label'):
                selected_folder = select_folder_dialog("Select Label Folder (Annotations)")
                if selected_folder:
                    st.session_state['label_folder'] = selected_folder
                st.session_state['browse_label'] = False
                st.rerun()
        
        col1, col2 = st.columns(2)
        
        # Get folder paths from session state (source of truth)
        dataset_folder = st.session_state.get('dataset_folder', '')
        label_folder = st.session_state.get('label_folder', '')
        
        with col1:
            st.markdown("**Dataset Folder (Images)**")
            
            # File uploader-style interface for folder selection
            if dataset_folder and os.path.exists(dataset_folder):
                st.success(f"Selected: {dataset_folder}")
            else:
                st.info("No folder selected. Click 'Browse files' to select a folder.")
            
            # Browse button styled like file uploader
            if TKINTER_AVAILABLE:
                if st.button("Browse dataset folder", key="browse_dataset_btn", width='stretch', type="primary"):
                    st.session_state['browse_dataset'] = True
                    st.rerun()
            else:
                # Fallback to text input if tkinter not available
                manual_dataset_folder = st.text_input(
                    "Enter folder path manually:",
                    value=dataset_folder,
                    help="Path to folder containing image files",
                    key="dataset_folder_input_manual"
                )
                if manual_dataset_folder != dataset_folder:
                    st.session_state['dataset_folder'] = manual_dataset_folder
                    dataset_folder = manual_dataset_folder
            
            # Show validation
            if dataset_folder:
                if os.path.exists(dataset_folder):
                    st.caption(f"✓ Valid folder: {os.path.basename(dataset_folder)}")
                else:
                    st.warning(f"⚠ Path not found: {dataset_folder}")
            
            file_extension = st.selectbox(
                "Image File Extension",
                options=["*.jpg", "*.png", "*.jpeg"],
                index=0
            )
        
        with col2:
            st.markdown("**Label Folder (Annotations)**")
            
            # File uploader-style interface for folder selection
            if label_folder and os.path.exists(label_folder):
                st.success(f"Selected: {label_folder}")
            else:
                st.info("No folder selected. Click 'Browse files' to select a folder.")
            
            # Browse button styled like file uploader
            if TKINTER_AVAILABLE:
                if st.button("Browse label folder", key="browse_label_btn", width='stretch', type="primary"):
                    st.session_state['browse_label'] = True
                    st.rerun()
            else:
                # Fallback to text input if tkinter not available
                manual_label_folder = st.text_input(
                    "Enter folder path manually:",
                    value=label_folder,
                    help="Path to folder containing YOLO format .txt label files",
                    key="label_folder_input_manual"
                )
                if manual_label_folder != label_folder:
                    st.session_state['label_folder'] = manual_label_folder
                    label_folder = manual_label_folder
            
            # Show validation
            if label_folder:
                if os.path.exists(label_folder):
                    st.caption(f"✓ Valid folder: {os.path.basename(label_folder)}")
                else:
                    st.warning(f"⚠ Path not found: {label_folder}")
            
            class_names_input = st.text_input(
                "Class Names (comma-separated)",
                value="circle,diamond,flower,heptagram,pentagon,square,triangle",
                help="Comma-separated list of class names in order"
            )
        
        if not TKINTER_AVAILABLE:
            st.info("💡 **Tip:** To use folder browser, install tkinter: `pip install tk` (Linux) or it's included with Python on Windows/Mac.")

    # Parse class names
    class_names = [name.strip() for name in class_names_input.split(",") if name.strip()]

    # Analysis selection
    st.markdown("### Select Analysis")
    
    analysis_options = {
        "Color Distribution": ("analyze_color_distribution", "image"),
        "Brightness & Contrast": ("analyze_brightness_contrast", "image"),
        "Blur Detection": ("analyze_blur", "image"),
        "Image Sizes": ("plot_image_sizes", "image"),
        "Class Distribution": ("plot_class_distribution", "label"),
        "Aspect Ratio": ("plot_aspect_ratio_separate", "label"),
        "Objects per Image": ("plot_objects_per_image", "label"),
    }

    selected_analyses = st.multiselect(
        "Choose analyses to run:",
        options=list(analysis_options.keys()),
        default=list(analysis_options.keys())
    )

    if st.button("🚀 Run EDA Analysis", type="primary"):
        if not selected_analyses:
            st.warning("Please select at least one analysis to run.")
            return

        # Get folder paths from session state (source of truth)
        # These are already set above, but ensure we use session state
        dataset_folder_final = st.session_state.get('dataset_folder', '')
        label_folder_final = st.session_state.get('label_folder', '')
        
        # Update session state with current values if they exist
        if dataset_folder and dataset_folder != dataset_folder_final:
            st.session_state['dataset_folder'] = dataset_folder
            dataset_folder_final = dataset_folder
        if label_folder and label_folder != label_folder_final:
            st.session_state['label_folder'] = label_folder
            label_folder_final = label_folder

        # Validate paths
        image_analyses = [name for name in selected_analyses 
                         if analysis_options[name][1] == "image"]
        label_analyses = [name for name in selected_analyses 
                         if analysis_options[name][1] == "label"]

        if image_analyses and not os.path.exists(dataset_folder_final):
            st.error(f"Dataset folder not found: {dataset_folder_final}")
            return

        if label_analyses and not os.path.exists(label_folder_final):
            st.error(f"Label folder not found: {label_folder_final}")
            return

        if label_analyses and not class_names:
            st.error("Please provide class names for label-based analyses.")
            return

        # Run selected analyses using final folder paths
        for analysis_name in selected_analyses:
            st.markdown(f"---")
            st.subheader(f"📊 {analysis_name}")

            try:
                # Create column layout for 80% width (10% margin on each side)
                col_left, col_center, col_right = st.columns([1, 8, 1])
                
                with col_center:
                    if analysis_name == "Color Distribution":
                        fig = analyze_color_distribution(dataset_folder_final, file_extension)
                        if fig:
                            st.pyplot(fig, width='stretch')
                            plt.close(fig)

                    elif analysis_name == "Brightness & Contrast":
                        fig = analyze_brightness_contrast(dataset_folder_final, file_extension)
                        if fig:
                            st.pyplot(fig, width='stretch')
                            plt.close(fig)

                    elif analysis_name == "Blur Detection":
                        fig = analyze_blur(dataset_folder_final, file_extension)
                        if fig:
                            st.pyplot(fig, width='stretch')
                            plt.close(fig)

                    elif analysis_name == "Image Sizes":
                        fig = plot_image_sizes(dataset_folder_final, file_extension)
                        if fig:
                            st.pyplot(fig, width='stretch')
                            plt.close(fig)

                    elif analysis_name == "Class Distribution":
                        fig = plot_class_distribution(label_folder_final, class_names)
                        if fig:
                            st.pyplot(fig, width='stretch')
                            plt.close(fig)

                    elif analysis_name == "Aspect Ratio":
                        fig = plot_aspect_ratio_separate(label_folder_final, class_names)
                        if fig:
                            st.pyplot(fig, width='stretch')
                            plt.close(fig)

                    elif analysis_name == "Objects per Image":
                        fig = plot_objects_per_image(label_folder_final)
                        if fig:
                            st.pyplot(fig, width='stretch')
                            plt.close(fig)

            except Exception as e:
                st.error(f"Error running {analysis_name}: {str(e)}")
                st.code(str(e))
                import traceback
                st.code(traceback.format_exc())

        st.success("✅ EDA Analysis Complete!")
