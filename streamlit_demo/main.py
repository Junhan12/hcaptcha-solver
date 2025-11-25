import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import base64
import time
import io

# Ensure project root is on sys.path to import client and app modules
_this_dir = os.path.dirname(__file__)
_project_root = os.path.abspath(os.path.join(_this_dir, '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from client.crawler import run_crawl_once
from app import decompress_image_to_base64
from app.config import API_TIMEOUT

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
    st.info("Configure and apply preprocessing steps to your dataset images.")
    st.write("🚧 **Feature coming soon** - This will allow you to configure preprocessing profiles and apply them to images.")
    
elif main_section == "4. Data Augmentation":
    st.header("Data Augmentation")
    st.info("Apply data augmentation techniques to expand your training dataset.")
    st.write("🚧 **Feature coming soon** - This will provide data augmentation options for training data expansion.")
    
elif main_section == "5. Create and Upload Model":
    # Use existing "Create/Update Model" functionality
    st.subheader("Create / Update Model")
    with st.form("model_form"):
        model_id = st.text_input("Model ID", placeholder="m-001")
        model_name = st.text_input("Model Name", placeholder="yolov8-object-001")
        weights = st.file_uploader("Weights (.pt)", type=["pt"])
        col1, col2, col3 = st.columns(3)
        with col1:
            precision = st.number_input("precision", min_value=0.0, max_value=1.0, step=0.0001, format="%0.4f")
            recall = st.number_input("recall", min_value=0.0, max_value=1.0, step=0.0001, format="%0.4f")
        with col2:
            f1_score = st.number_input("f1_score", min_value=0.0, max_value=1.0, step=0.0001, format="%0.4f")
            mAP50 = st.number_input("mAP50", min_value=0.0, max_value=1.0, step=0.0001, format="%0.4f")
        with col3:
            AP5095 = st.number_input("AP5095", min_value=0.0, max_value=1.0, step=0.0001, format="%0.4f")
            is_active = st.checkbox("Set Active", value=False)
        submitted = st.form_submit_button("Save Model")

    if submitted:
        if not model_id or not model_name:
            st.error("model_id and model_name are required")
        else:
            data = {
                "model_id": model_id,
                "model_name": model_name,
                "precision": str(precision),
                "recall": str(recall),
                "f1_score": str(f1_score),
                "mAP50": str(mAP50),
                "AP5095": str(AP5095),
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
    st.info("View and evaluate model training metrics and performance.")
    st.write("🚧 **Feature coming soon** - This will display training history, metrics, and model evaluation results.")
    
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
                                
                                st.image(img, caption="Processed Image with Detections", use_container_width=True)
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
                            steps = postprocess.get('steps', {})
                            conf_thresh = steps.get('confidence_threshold', 'N/A')
                            iou_thresh = steps.get('iou_threshold', 'N/A')
                            st.info(f"**Postprocessing:** {postprocess.get('name', 'N/A')}")
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
                        st.dataframe(table_data, use_container_width=True)
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
                                    
                                    if processed_img_b64:
                                        try:
                                            img_bytes = base64.b64decode(processed_img_b64)
                                            img = Image.open(io.BytesIO(img_bytes))
                                            
                                            # Draw bounding boxes
                                            if results and isinstance(results, list):
                                                draw = ImageDraw.Draw(img)
                                                try:
                                                    font = ImageFont.truetype("arial.ttf", 16)
                                                except:
                                                    try:
                                                        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                                                    except:
                                                        font = ImageFont.load_default()
                                                
                                                for result_item in results:
                                                    if isinstance(result_item, dict) and 'error' not in result_item:
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
                                            
                                            st.image(img, caption=f"{filename} (Challenge {challenge_idx})", use_container_width=True)
                                        except Exception as e:
                                            st.error(f"Failed to display image: {e}")
                                    elif data_url:
                                        try:
                                            b64_part = data_url.split(",", 1)[1] if "," in data_url else data_url
                                            img_bytes = base64.b64decode(b64_part)
                                            st.image(img_bytes, caption=f"{filename}", use_container_width=True)
                                        except Exception:
                                            st.write(f"{filename}")
                                
                                with col2:
                                    if isinstance(result, dict):
                                        preprocess = result.get("preprocess")
                                        if preprocess:
                                            st.info(f"**Preprocessing:** {preprocess.get('preprocess_id', 'N/A')}")
                                        
                                        postprocess = result.get("postprocess")
                                        if postprocess:
                                            steps = postprocess.get('steps', {})
                                            conf_thresh = steps.get('confidence_threshold', 'N/A')
                                            iou_thresh = steps.get('iou_threshold', 'N/A')
                                            st.info(f"**Postprocessing:** {postprocess.get('name', 'N/A')}")
                                            st.caption(f"Conf: {conf_thresh}, IoU: {iou_thresh}")
                                        
                                        if 'perform_time' in result:
                                            st.metric("Processing Time", f"{result.get('perform_time', 0):.3f}s")
                                        
                                        if 'challenge_id' in result:
                                            st.caption(f"**Challenge ID:** {result.get('challenge_id')}")
                                        
                                        if 'error' in result:
                                            st.error(f"**Error:** {result.get('error')}")
                                        elif results:
                                            if isinstance(results, list) and len(results) > 0:
                                                st.success(f"**Detections:** {len(results)} objects")
                                            else:
                                                st.info("No detections found")
                                
                                # Display results table
                                if results and isinstance(results, list) and len(results) > 0:
                                    st.markdown(f"**Detection Results for {filename}:**")
                                    table_data = []
                                    for result_idx, result_item in enumerate(results):
                                        if isinstance(result_item, dict) and 'error' not in result_item:
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
                                        st.dataframe(table_data, use_container_width=True)
                                
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
                                    steps = postprocess.get('steps', {})
                                    conf_thresh = steps.get('confidence_threshold', 'N/A')
                                    iou_thresh = steps.get('iou_threshold', 'N/A')
                                    st.info(f"**Postprocessing:** {postprocess.get('name', 'N/A')}")
                                    st.caption(f"Conf: {conf_thresh}, IoU: {iou_thresh}")
                        
                        # Display each image with results
                        batch_results = first_result.get('results', []) if isinstance(first_result, dict) else []
                        processed_images = first_result.get('processed_images', []) if isinstance(first_result, dict) else []
                        
                        # Create a mapping from image_index to batch result for easier lookup
                        batch_results_by_index = {}
                        for batch_item in batch_results:
                            if isinstance(batch_item, dict) and 'image_index' in batch_item:
                                batch_results_by_index[batch_item['image_index']] = batch_item
                        
                        # Detect if first item is a sample tile (sent separately, not in batch)
                        has_sample_tile = False
                        if len(accepted) == len(batch_results) + 1 and len(accepted) > 0:
                            first_item_result = accepted[0].get("result", {})
                            if isinstance(first_item_result, dict):
                                first_results = first_item_result.get('results', [])
                                if isinstance(first_results, list) and len(first_results) > 0:
                                    if not isinstance(first_results[0], dict) or 'image_index' not in first_results[0]:
                                        has_sample_tile = True
                                elif isinstance(first_item_result, dict) and 'processed_image' in first_item_result:
                                    has_sample_tile = True
                        
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
                                image_result_data = batch_results_by_index.get(image_index)
                                if image_result_data is None and batch_idx >= 0 and batch_idx < len(batch_results):
                                    image_result_data = batch_results[batch_idx]
                            
                            st.markdown(f"#### {filename}")
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                display_img_b64 = None
                                if has_sample_tile and img_idx == 0:
                                    if isinstance(result, dict) and result.get('processed_image'):
                                        display_img_b64 = result.get('processed_image')
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
                                        
                                        st.image(img, caption=f"{filename} (with detections)", use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Failed to display image: {e}")
                                else:
                                    if data_url:
                                        try:
                                            b64_part = data_url.split(",", 1)[1] if "," in data_url else data_url
                                            img_bytes = base64.b64decode(b64_part)
                                            st.image(img_bytes, caption=f"{filename}", use_container_width=True)
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
                                table_display_result = result
                            elif image_result_data:
                                table_display_result = image_result_data
                            
                            if table_display_result:
                                img_results = table_display_result.get('results', [])
                                if isinstance(img_results, list) and len(img_results) > 0:
                                    valid_detections = [r for r in img_results if isinstance(r, dict) and 'error' not in r and 'bbox' in r]
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
                                            st.dataframe(table_data, use_container_width=True)
                                    else:
                                        st.info(f"No valid detections for {filename}")
                                elif isinstance(img_results, dict) and 'error' in img_results:
                                    st.error(f"Error: {img_results.get('error')}")
                                else:
                                    st.info(f"No detections found for {filename}")
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
                                
                                if processed_img_b64:
                                    try:
                                        img_bytes = base64.b64decode(processed_img_b64)
                                        img = Image.open(io.BytesIO(img_bytes))
                                        
                                        if results and isinstance(results, list):
                                            draw = ImageDraw.Draw(img)
                                            try:
                                                font = ImageFont.truetype("arial.ttf", 16)
                                            except:
                                                try:
                                                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                                                except:
                                                    font = ImageFont.load_default()
                                            
                                            for result_item in results:
                                                if isinstance(result_item, dict) and 'error' not in result_item:
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
                                        
                                        st.image(img, caption="Processed Image with Detections", use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Failed to display image: {e}")
                                else:
                                    if data_url:
                                        try:
                                            b64_part = data_url.split(",", 1)[1] if "," in data_url else data_url
                                            img_bytes = base64.b64decode(b64_part)
                                            st.image(img_bytes, caption=f"{filename}", use_container_width=True)
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
                                        steps = postprocess.get('steps', {})
                                        conf_thresh = steps.get('confidence_threshold', 'N/A')
                                        iou_thresh = steps.get('iou_threshold', 'N/A')
                                        st.info(f"**Postprocessing:** {postprocess.get('name', 'N/A')}")
                                        st.caption(f"Conf: {conf_thresh}, IoU: {iou_thresh}")
                                    
                                    if 'perform_time' in result:
                                        st.metric("Processing Time", f"{result.get('perform_time', 0):.3f}s")
                                    
                                    if 'challenge_id' in result:
                                        st.caption(f"**Challenge ID:** {result.get('challenge_id')}")
                                    
                                    if 'error' in result:
                                        st.error(f"**Error:** {result.get('error')}")
                                        st.caption("X Inference NOT saved (errors are not stored)")
                                    elif results:
                                        if isinstance(results, list) and len(results) > 0:
                                            st.success(f"**Detections:** {len(results)} objects")
                                            st.caption("/ Inference saved to database")
                                        elif isinstance(results, dict) and 'message' in results:
                                            st.info(f"**Message:** {results.get('message')}")
                                            st.caption("/ Inference saved to database")
                                        else:
                                            st.info("No detections found")
                                            st.caption("/ Inference saved to database")
                            
                            if results and isinstance(results, list) and len(results) > 0:
                                st.markdown("### Detection Results")
                                table_data = []
                                for result_idx, result_item in enumerate(results):
                                    if isinstance(result_item, dict) and 'error' not in result_item:
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
                                    st.dataframe(table_data, use_container_width=True)
                            else:
                                st.info("No detections found")
                            
                            with st.expander("View Full Response"):
                                st.json(result)
                            
                            if idx < len(accepted) - 1:
                                st.markdown("---")
            else:
                st.warning("Crawl finished but no images were sent.")
            st.info(f"Elapsed: {elapsed:.2f}s")
            progress.progress(100)