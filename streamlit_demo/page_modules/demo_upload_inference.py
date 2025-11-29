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

from utils import API_TIMEOUT

def render(progress, status):
    """Render the Upload Image for Inference page."""
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

                            # Get processed image dimensions
                            processed_width, processed_height = img.size

                            # Draw bounding boxes on image if there are detections
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

                                caption = f"Processed Image with Detections ({processed_width} × {processed_height})"
                            else:
                                caption = f"Processed Image - No Detections ({processed_width} × {processed_height})"

                            # Always display the processed image at its actual size
                            st.image(img, caption=caption, width=processed_width)
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


