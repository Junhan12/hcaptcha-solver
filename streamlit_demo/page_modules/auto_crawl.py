"""
Auto Crawl Dataset page.
"""
import streamlit as st
import os
import sys
import base64
import re
import time
import traceback
from io import BytesIO
from pathlib import Path

# Path setup
_this_dir = os.path.dirname(__file__)
_parent_dir = os.path.abspath(os.path.join(_this_dir, '..'))
_project_root = os.path.abspath(os.path.join(_this_dir, '..', '..'))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import crawler utilities
try:
    from client.crawler import (
        get_challenge_type_id_for_question,
        check_question_matches_challenge_type,
        retrieve_question,
        click_refresh_button,
        check_submit_button_exists,
        click_submit_button,
        fetch_image_bytes,
    )
    from app.database import _find_challenge_type_for_question
    CRAWLER_AVAILABLE = True
except ImportError as e:
    CRAWLER_AVAILABLE = False
    st.warning(f"Could not import crawler modules: {e}")

# Try to import tkinter for folder selection dialog
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

# Selenium imports
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


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


def get_next_image_number(folder_path):
    """
    Get the next image number by checking existing files in the folder.
    Looks for files named 'image1.jpg', 'image2.jpg', etc. and returns the next number.
    
    Args:
        folder_path: Path to the folder containing images
    
    Returns:
        int: Next image number (e.g., if image3.jpg exists, returns 4)
    """
    if not os.path.exists(folder_path):
        return 1
    
    max_num = 0
    pattern = re.compile(r'^image(\d+)\.(jpg|jpeg|png)$', re.IGNORECASE)
    
    try:
        for filename in os.listdir(folder_path):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)
    except Exception:
        pass
    
    return max_num + 1


def get_next_sample_number(folder_path):
    """
    Get the next sample image number by checking existing sample files in the folder.
    Looks for files named 'sample1.jpg', 'sample2.jpg', etc. and returns the next number.
    
    Args:
        folder_path: Path to the folder containing images
    
    Returns:
        int: Next sample number (e.g., if sample3.jpg exists, returns 4)
    """
    if not os.path.exists(folder_path):
        return 1
    
    max_num = 0
    pattern = re.compile(r'^sample(\d+)\.(jpg|jpeg|png)$', re.IGNORECASE)
    
    try:
        for filename in os.listdir(folder_path):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)
    except Exception:
        pass
    
    return max_num + 1


def save_canvas_images_locally(driver, challenge_type_id, output_dir, question):
    """
    Extract canvas images and save them locally to challenge_type folder.
    
    Args:
        driver: Selenium WebDriver instance
        challenge_type_id: Challenge type ID (e.g., 'ct-001')
        output_dir: Base output directory
        question: Question text for validation
    
    Returns:
        tuple: (saved_count, saved_files)
    """
    if not challenge_type_id:
        return 0, []
    
    canvases = driver.find_elements(By.TAG_NAME, "canvas")
    if not canvases:
        return 0, []
    
    # Create challenge_type folder
    challenge_folder = os.path.join(output_dir, challenge_type_id)
    os.makedirs(challenge_folder, exist_ok=True)
    
    saved_count = 0
    saved_files = []
    
    # Get starting image number (check folder once before loop)
    current_num = get_next_image_number(challenge_folder)
    
    for i, canvas in enumerate(canvases, start=1):
        try:
            # Extract canvas image as base64
            b64_payload = driver.execute_script(
                "return arguments[0].toDataURL('image/png').substring(22);", canvas)
            image_data = base64.b64decode(b64_payload)
            
            # Use current number and increment for next image
            filename = f"image{current_num}.png"
            file_path = os.path.join(challenge_folder, filename)
            
            # Save image
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            saved_files.append({
                "filename": filename,
                "path": file_path,
                "challenge_type": challenge_type_id
            })
            saved_count += 1
            current_num += 1  # Increment for next image
            
        except Exception as e:
            # Log error but continue processing
            print(f"Failed to save canvas {i}: {e}")
    
    return saved_count, saved_files


def save_div_images_locally(driver, challenge_type_id, output_dir, question):
    """
    Extract div images and save them locally to challenge_type folder.
    
    Args:
        driver: Selenium WebDriver instance
        challenge_type_id: Challenge type ID (e.g., 'ct-001')
        output_dir: Base output directory
        question: Question text for validation
    
    Returns:
        tuple: (saved_count, saved_files)
    """
    if not challenge_type_id:
        return 0, []
    
    divs = driver.find_elements(By.XPATH, "//div[contains(@class, 'image')]")
    if not divs:
        return 0, []
    
    # Create challenge_type folder
    challenge_folder = os.path.join(output_dir, challenge_type_id)
    os.makedirs(challenge_folder, exist_ok=True)
    
    # Collect all image tiles
    image_list = []
    for div in divs:
        style = div.get_attribute("style")
        match = re.search(r'url\("(.*?)"\)', style)
        if match:
            url = match.group(1)
            img_bytes = fetch_image_bytes(url)
            if img_bytes is not None:
                image_list.append(img_bytes)
    
    if not image_list:
        return 0, []
    
    # Handle 10-tile case (first is sample/reference image, rest are clickable tiles)
    total_tiles = len(image_list)
    sample_image = None
    batch_images = image_list
    
    if total_tiles == 10:
        # First image is the sample/reference image
        sample_image = image_list[0]
        batch_images = image_list[1:]  # Rest are clickable tiles
    
    saved_count = 0
    saved_files = []
    
    # Save sample image first if it exists
    if sample_image:
        try:
            # Get next sample number
            sample_num = get_next_sample_number(challenge_folder)
            filename = f"sample{sample_num}.jpg"
            file_path = os.path.join(challenge_folder, filename)
            
            # Save sample image
            with open(file_path, 'wb') as f:
                f.write(sample_image)
            
            saved_files.append({
                "filename": filename,
                "path": file_path,
                "challenge_type": challenge_type_id,
                "type": "sample"
            })
            saved_count += 1
        except Exception as e:
            print(f"Failed to save sample image: {e}")
    
    # Get starting image number for batch tiles (check folder once before loop)
    current_num = get_next_image_number(challenge_folder)
    
    # Save batch tiles
    for img_bytes in batch_images:
        try:
            # Use current number and increment for next image
            filename = f"image{current_num}.jpg"
            file_path = os.path.join(challenge_folder, filename)
            
            # Save image
            with open(file_path, 'wb') as f:
                f.write(img_bytes)
            
            saved_files.append({
                "filename": filename,
                "path": file_path,
                "challenge_type": challenge_type_id,
                "type": "tile"
            })
            saved_count += 1
            current_num += 1  # Increment for next image
            
        except Exception as e:
            # Log error but continue processing
            print(f"Failed to save div image: {e}")
    
    return saved_count, saved_files


def run_crawl_and_save_locally(output_dir, max_rounds=20):
    """
    Run crawler and save images locally instead of sending to API.
    
    Args:
        output_dir: Directory to save crawled images
        max_rounds: Maximum number of challenge rounds to process
    
    Returns:
        dict: Summary of crawl results
    """
    if not SELENIUM_AVAILABLE:
        raise ImportError("Selenium is required for crawling")
    
    if not CRAWLER_AVAILABLE:
        raise ImportError("Crawler modules are not available")
    
    driver = None
    summary = {
        "total_images_saved": 0,
        "challenge_types": {},
        "questions": [],
        "errors": []
    }
    
    try:
        # Setup Chrome driver
        options = Options()
        options.add_argument("--start-maximized")
        options.add_experimental_option("detach", True)
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        try:
            driver.set_window_position(0, 0)
            driver.set_window_size(1400, 900)
        except Exception:
            pass
        
        # Navigate to hCaptcha demo
        driver.get("https://accounts.hcaptcha.com/demo")
        
        # Click checkbox
        iframe = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//iframe[@title='Widget containing checkbox for hCaptcha security challenge']"))
        )
        driver.switch_to.frame(iframe)
        checkbox = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "checkbox")))
        checkbox.click()
        driver.switch_to.default_content()
        
        # Switch to challenge iframe
        challenge_iframe = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//iframe[contains(@title, 'hCaptcha challenge')]"))
        )
        driver.switch_to.frame(challenge_iframe)
        
        # Main processing loop
        round_number = 0
        
        while round_number < max_rounds:
            round_number += 1
            
            # Retrieve question
            prompt_text = retrieve_question(driver, timeout=10)
            if not prompt_text:
                break
            
            # Validate question matches a challenge_type
            matched = check_question_matches_challenge_type(prompt_text)
            if not matched:
                # Try to refresh
                if click_refresh_button(driver, attempt_number=round_number):
                    time.sleep(3)
                    continue
                else:
                    continue
            
            # Get challenge_type_id
            challenge_type_id = get_challenge_type_id_for_question(prompt_text)
            if not challenge_type_id:
                error_msg = f"Could not determine challenge_type_id for question: {prompt_text}"
                summary["errors"].append(error_msg)
                print(f"Warning: {error_msg}")
                if click_refresh_button(driver, attempt_number=round_number):
                    time.sleep(3)
                    continue
                else:
                    continue
            
            summary["questions"].append({
                "round": round_number,
                "question": prompt_text,
                "challenge_type_id": challenge_type_id
            })
            
            # Track challenge types
            if challenge_type_id not in summary["challenge_types"]:
                summary["challenge_types"][challenge_type_id] = 0
            
            time.sleep(2)
            
            # Extract and save images
            canvases = driver.find_elements(By.TAG_NAME, "canvas")
            if canvases:
                saved_count, saved_files = save_canvas_images_locally(
                    driver, challenge_type_id, output_dir, prompt_text
                )
                summary["challenge_types"][challenge_type_id] += saved_count
                summary["total_images_saved"] += saved_count
            else:
                saved_count, saved_files = save_div_images_locally(
                    driver, challenge_type_id, output_dir, prompt_text
                )
                summary["challenge_types"][challenge_type_id] += saved_count
                summary["total_images_saved"] += saved_count
            
            # Check if submit button exists
            if check_submit_button_exists(driver, timeout=2):
                if click_submit_button(driver, wait_after_click=3):
                    time.sleep(2)
                    continue
                else:
                    break
            else:
                break
        
        if round_number >= max_rounds:
            warning_msg = f"Reached maximum rounds limit ({max_rounds}. Stopped data crawling.)"
            summary["errors"].append(warning_msg)
            print(f"Warning: {warning_msg}")
        
    except Exception as e:
        error_msg = f"Error during crawling: {e}"
        summary["errors"].append(error_msg)
        summary["errors"].append(traceback.format_exc())
        print(f"Error: {error_msg}")
    finally:
        if driver is not None:
            time.sleep(2)
            driver.quit()
    
    return summary


def render():
    """Render the Auto Crawl Dataset page."""
    # Add CSS to make buttons full-width
    st.markdown("""
        <style>
        div[data-testid="stButton"] > button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.header("Auto Crawl Dataset")
    st.info("Automatically crawl hCAPTCHA challenges and save images to local directory organized by challenge type.")
    
    if not SELENIUM_AVAILABLE:
        st.error("Selenium is not available. Please install: pip install selenium webdriver-manager")
        return
    
    if not CRAWLER_AVAILABLE:
        st.error("Crawler modules are not available. Please check the imports.")
        return
    
    # Initialize session state for output directory
    if 'crawl_output_folder' not in st.session_state:
        st.session_state['crawl_output_folder'] = ""
    
    # Handle folder selection BEFORE creating widgets
    if TKINTER_AVAILABLE:
        if 'browse_crawl_output' in st.session_state and st.session_state.get('browse_crawl_output'):
            selected_folder = select_folder_dialog("Select Output Directory for Crawled Images")
            if selected_folder:
                st.session_state['crawl_output_folder'] = selected_folder
            st.session_state['browse_crawl_output'] = False
            st.rerun()
    
    # Output directory selection
    st.markdown("### Output Directory Configuration")
    with st.expander("Folder Configuration", expanded=True):
        output_folder = st.session_state.get('crawl_output_folder', '')
        
        if output_folder and os.path.exists(output_folder):
            st.success(f"Selected: {output_folder}")
        else:
            st.info("No folder selected. Click 'Browse files' to select output directory.")
        
        if TKINTER_AVAILABLE:
            if st.button("Browse files", key="browse_crawl_output_btn", type="primary"):
                st.session_state['browse_crawl_output'] = True
                st.rerun()
        else:
            manual_output_folder = st.text_input(
                "Enter output directory path manually:",
                value=output_folder,
                help="Path to directory where crawled images will be saved (organized by challenge_type)",
                key="crawl_output_folder_input_manual"
            )
            if manual_output_folder != output_folder:
                st.session_state['crawl_output_folder'] = manual_output_folder
                output_folder = manual_output_folder
        
        if output_folder:
            if os.path.exists(output_folder):
                st.caption(f"Valid folder: {os.path.basename(output_folder)}")
            else:
                st.warning(f"Path not found")
        
        # Max rounds configuration
        max_rounds = st.number_input(
            "Maximum Challenge Rounds",
            min_value=1,
            max_value=50,
            value=20,
            help="Maximum number of challenge rounds to process"
        )
    
    st.markdown("---")
    
    # Start crawling button
    if st.button("Start Crawling", key="start_crawl_button", type="primary"):
        output_folder = st.session_state.get('crawl_output_folder', '')
        
        # Validate inputs
        if not output_folder:
            st.error("Please select an output directory.")
        elif not os.path.exists(output_folder):
            st.error(f"Output directory not found: {output_folder}")
        else:
            # Status container
            status_text = st.empty()
            status_text.info("Starting crawler... This may take a few minutes.")
            
            try:
                # Run crawler
                with st.spinner("Crawling hCAPTCHA challenges..."):
                    summary = run_crawl_and_save_locally(output_folder, max_rounds=max_rounds)
                
                # Display results
                status_text.success("Crawling Complete!")
                
                st.markdown("### Crawl Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Images Saved", summary["total_images_saved"])
                with col2:
                    st.metric("Challenge Types Found", len(summary["challenge_types"]))
                
                # Display challenge types breakdown
                if summary["challenge_types"]:
                    st.markdown("#### Images by Challenge Type")
                    challenge_data = []
                    for ct_id, count in summary["challenge_types"].items():
                        challenge_data.append({
                            "Challenge Type": ct_id,
                            "Images Saved": count
                        })
                    
                    import pandas as pd
                    df_challenges = pd.DataFrame(challenge_data)
                    st.dataframe(df_challenges, width='stretch', hide_index=True)
                
                # Display questions encountered
                if summary["questions"]:
                    st.markdown("#### Questions Encountered")
                    for q_info in summary["questions"]:
                        with st.expander(f"Round {q_info['round']}: {q_info['question']}"):
                            st.caption(f"Challenge Type: {q_info['challenge_type_id']}")
                
                # Show output location
                st.info(f"**Output Location:** `{output_folder}`")
                st.caption("Images are organized in subfolders by challenge_type (e.g., ct-001/, ct-002/)")
                
            except Exception as e:
                status_text.error(f"Error during crawling: {e}")
                st.code(traceback.format_exc())
    
    st.markdown("---")
    
    # Information section
    st.markdown("### How It Works")
    st.markdown("""
    1. **Validation**: The crawler validates keywords and finds the challenge type from the question
    2. **Image Extraction**: Extracts images from canvas elements or div tiles
    3. **Local Storage**: Saves images to local directory organized by challenge_type (e.g., ct-001/)
    4. **Auto-increment**: Automatically increments filenames (image1.jpg, image2.jpg, etc.) based on existing files
    5. **No Inference**: Images are saved locally without inference or MongoDB storage
    """)
