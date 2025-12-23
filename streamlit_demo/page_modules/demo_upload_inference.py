"""
Upload Image for Inference demo page.
"""
import streamlit as st
import requests
import time
import base64
import io
from PIL import Image, ImageDraw, ImageFont

import sys
import os
_this_dir = os.path.dirname(__file__)
_parent_dir = os.path.abspath(os.path.join(_this_dir, '..'))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from utils import API_TIMEOUT, extract_detections

def render(progress, status):
    """Render the Upload Image for Inference page."""
    
    # Page description and overview
    st.header("Upload Image for Inference")
    
    st.markdown("""
    **Purpose**: Test the hCAPTCHA solver by uploading individual images and running inference through the API.
            
    **Process**: 
    1. Upload an hCAPTCHA challenge image (JPG, JPEG, PNG)
    2. Enter the challenge question text
    3. Click "Send For Inference" to process via API
    4. View results: original image, preprocessed image with bounding boxes, detection table, and metadata (model, preprocessing, postprocessing, timing)
    """)
    
    st.markdown("---")
    
    uploaded_img = st.file_uploader("Upload hCAPTCHA Image", type=["jpg", "jpeg", "png"])
    question = st.text_input("Enter hCAPTCHA Question")
    if st.button("Send For Inference"):
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
                
                # Handle error responses
                if out.get("error"):
                    st.error(f"Error: {out.get('error')}")
                    progress.progress(0)
                    return
                
                st.success("Uploaded and processed by API")

                # Extract detections using standard schema
                detections = extract_detections(out)
                
                # Display images in 1x2 layout (original and preprocessed with bounding boxes)
                if uploaded_img:
                    # Reset file pointer and load original image
                    uploaded_img.seek(0)
                    original_img = Image.open(uploaded_img)
                    original_width, original_height = original_img.size
                    
                    # Get preprocessed image from API response if available
                    processed_img = None
                    processed_width, processed_height = original_width, original_height
                    
                    processed_image_b64 = out.get("processed_image")
                    if processed_image_b64:
                        try:
                            processed_img_bytes = base64.b64decode(processed_image_b64)
                            processed_img = Image.open(io.BytesIO(processed_img_bytes))
                            processed_width, processed_height = processed_img.size
                        except Exception as e:
                            st.warning(f"Failed to decode preprocessed image: {e}")
                            # Fallback to original image
                            processed_img = original_img.copy()
                    else:
                        # No preprocessing was applied, use original
                        processed_img = original_img.copy()
                    
                    # Draw bounding boxes on preprocessed image if there are detections
                    if detections and len(detections) > 0:
                        draw = ImageDraw.Draw(processed_img)
                        # Try to load a font, fallback to default if not available
                        try:
                            font = ImageFont.truetype("arial.ttf", 16)
                        except:
                            try:
                                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                            except:
                                font = ImageFont.load_default()

                        for detection in detections:
                            # Ensure detection is a dictionary with required fields
                            if not isinstance(detection, dict):
                                continue
                            bbox = detection.get('bbox', [])
                            if len(bbox) >= 4:
                                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                                # Draw rectangle
                                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                                # Draw label
                                class_name = detection.get('class', 'Unknown')
                                confidence = detection.get('confidence', 0.0)
                                label = f"{class_name}: {confidence:.2f}"
                                # Get text size for background
                                bbox_text = draw.textbbox((0, 0), label, font=font)
                                text_width = bbox_text[2] - bbox_text[0]
                                text_height = bbox_text[3] - bbox_text[1]
                                # Draw text background
                                draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill="red")
                                # Draw text
                                draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=font)

                    # Display images in 1x2 layout
                    img_col1, img_col2 = st.columns(2)
                    
                    with img_col1:
                        st.image(original_img, caption=f"Original Image ({original_width} × {original_height})", width='stretch')
                    
                    with img_col2:
                        if detections and len(detections) > 0:
                            caption = f"Preprocessed Image with Detections ({processed_width} × {processed_height})"
                        else:
                            caption = f"Preprocessed Image - No Detections ({processed_width} × {processed_height})"
                        st.image(processed_img, caption=caption, width='stretch')
                else:
                    st.warning("No image available")

                # Display inference information in 1x4 layout
                st.markdown("---")
                info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                
                with info_col1:
                    # Display model info
                    model = out.get("model")
                    if model:
                        model_id = model.get('model_id', 'N/A')
                        model_name = model.get('model_name', 'N/A')
                        st.info(f"**Model ID:** {model_id}")
                        st.caption(f"Model Name: {model_name}")
                    else:
                        st.info("**Model ID:** N/A")
                        st.caption("Model Name: N/A")
                
                with info_col2:
                    # Display preprocessing applied
                    preprocess = out.get("preprocess")
                    if preprocess:
                        preprocess_id = preprocess.get('preprocess_id', 'N/A')
                        applied_steps = preprocess.get('applied_steps', [])
                        st.info(f"**Preprocessing Applied:** {preprocess_id}")
                        if applied_steps and isinstance(applied_steps, list) and len(applied_steps) > 0:
                            step_names = [step.get('operation', '') for step in applied_steps if isinstance(step, dict)]
                            if step_names:
                                st.caption(f"Steps: {', '.join(step_names)}")
                            else:
                                st.caption("No preprocessing steps")
                        else:
                            st.caption("No preprocessing steps")
                    else:
                        st.info("**Preprocessing Applied:** N/A")
                        st.caption("No preprocessing applied")
                
                with info_col3:
                    # Display postprocessing applied
                    postprocess = out.get("postprocess")
                    if postprocess:
                        postprocess_name = postprocess.get('name', 'N/A')
                        postprocess_id = postprocess.get('postprocess_id', 'N/A')
                        st.info(f"**Postprocessing Applied:** {postprocess_id}")
                        if postprocess_name and postprocess_name != 'N/A':
                            st.caption(f"Name: {postprocess_name}")
                        
                        steps = postprocess.get('steps', [])
                        if isinstance(steps, list) and len(steps) > 0:
                            operations = [step.get('operation', 'Unknown') for step in steps if isinstance(step, dict)]
                            if operations:
                                st.caption(f"Operations: {', '.join(operations)}")
                            else:
                                st.caption("No operations")
                        else:
                            st.caption("No operations")
                    else:
                        st.info("**Postprocessing Applied:** N/A")
                        st.caption("No postprocessing applied")
                
                with info_col4:
                    # Display performance
                    perform_time = out.get('perform_time')
                    if perform_time:
                        st.metric("Processing Time", f"{perform_time:.3f}s")
                    else:
                        st.metric("Processing Time", "N/A")

                # Display results table using standard schema
                if detections and len(detections) > 0:
                    st.markdown("### Detection Results")
                    # Prepare data for table
                    table_data = []
                    for idx, detection in enumerate(detections):
                        # Ensure detection is a dictionary with required fields (bbox, class, confidence)
                        if not isinstance(detection, dict):
                            continue
                        bbox = detection.get('bbox', [])
                        if len(bbox) < 4:
                            continue  # Skip invalid detections
                        bbox_str = f"[{', '.join([str(int(x)) for x in bbox[:4]])}]"
                        table_data.append({
                            "ID": idx + 1,
                            "Class": detection.get('class', 'Unknown'),
                            "Confidence": f"{detection.get('confidence', 0.0):.4f}",
                            "Bounding Box": bbox_str,
                            "Coordinates": f"({bbox[0]:.1f}, {bbox[1]:.1f}) to ({bbox[2]:.1f}, {bbox[3]:.1f})"
                        })
                    if table_data:
                        st.dataframe(table_data, width='stretch')
                    else:
                        st.info("No valid detections found")
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


