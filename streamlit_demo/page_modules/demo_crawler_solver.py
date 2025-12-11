"""
Auto Crawler, Solver, and Clicker demo page.
"""
import streamlit as st
import time
import base64
import io
import os
from PIL import Image, ImageDraw, ImageFont

import sys
import os
_this_dir = os.path.dirname(__file__)
_parent_dir = os.path.abspath(os.path.join(_this_dir, '..'))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from utils import (
    decompress_image_to_base64,
    extract_detections,
)
# Import run_crawl_once directly from source to avoid Streamlit caching issues
from client.crawler import run_crawl_once

def render(progress, status):
    """Render the Auto Crawler, Solver, and Clicker page."""
    # Allow user to select number of rounds
    st.markdown("### Configuration")
    max_rounds = st.number_input(
        "Number of Rounds to Process",
        min_value=1,
        max_value=50,
        value=20,
        help="Maximum number of challenge rounds to process. The crawler will stop earlier if the challenge is completed or if no submit button is found."
    )
    
    st.markdown("---")
    
    # Use existing "Crawl -> API" functionality
    if st.button("Crawl Demo Site And Solve Challenges", key="crawl_demo_button"):
        status.info(f"Launching browser and crawling demo site (max {max_rounds} rounds)...")
        progress.progress(10)
        start = time.time()
        summary = run_crawl_once(max_rounds=max_rounds)
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

                            # Extract detections using standard schema
                            detections = extract_detections(result) if isinstance(result, dict) else []
                            
                            # Display images in 1x2 layout (original and preprocessed)
                            original_img = None
                            preprocessed_img = None
                            
                            # Get original image from data_url
                            if data_url:
                                try:
                                    b64_part = data_url.split(",", 1)[1] if "," in data_url else data_url
                                    original_img_bytes = base64.b64decode(b64_part)
                                    original_img = Image.open(io.BytesIO(original_img_bytes))
                                except Exception as e:
                                    st.warning(f"Failed to load original image: {e}")
                            
                            # Get preprocessed image from API response
                            processed_image_b64 = result.get("processed_image") if isinstance(result, dict) else None
                            if processed_image_b64:
                                try:
                                    processed_img_bytes = base64.b64decode(processed_image_b64)
                                    preprocessed_img = Image.open(io.BytesIO(processed_img_bytes))
                                except Exception as e:
                                    st.warning(f"Failed to decode preprocessed image: {e}")
                                    # Fallback to original if available
                                    if original_img:
                                        preprocessed_img = original_img.copy()
                            elif original_img:
                                # No preprocessing was applied, use original
                                preprocessed_img = original_img.copy()
                            
                            # Draw bounding boxes on preprocessed image if there are detections
                            if preprocessed_img and detections and len(detections) > 0:
                                # Filter for valid detections (must have bbox with 4 elements)
                                valid_detections = [
                                    d for d in detections 
                                    if isinstance(d, dict) 
                                    and 'bbox' in d 
                                    and isinstance(d.get('bbox'), (list, tuple))
                                    and len(d.get('bbox', [])) >= 4
                                    and all(isinstance(x, (int, float)) for x in d.get('bbox', [])[:4])
                                ]

                                if valid_detections:
                                    draw = ImageDraw.Draw(preprocessed_img)
                                    try:
                                        font = ImageFont.truetype("arial.ttf", 16)
                                    except:
                                        try:
                                            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                                        except:
                                            font = ImageFont.load_default()

                                    for detection in valid_detections:
                                        bbox = detection.get('bbox', [])
                                        if len(bbox) >= 4:
                                            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                                            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                                            class_name = detection.get('class', 'Unknown')
                                            confidence = detection.get('confidence', 0.0)
                                            label = f"{class_name}: {confidence:.2f}"
                                            bbox_text = draw.textbbox((0, 0), label, font=font)
                                            text_width = bbox_text[2] - bbox_text[0]
                                            text_height = bbox_text[3] - bbox_text[1]
                                            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill="red")
                                            draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=font)
                            
                            # Display images in 1x2 layout
                            if original_img and preprocessed_img:
                                img_col1, img_col2 = st.columns(2)
                                
                                original_width, original_height = original_img.size
                                processed_width, processed_height = preprocessed_img.size
                                
                                with img_col1:
                                    st.image(original_img, caption=f"Original Image ({original_width} × {original_height})", width='stretch')
                                
                                with img_col2:
                                    if detections and len(detections) > 0:
                                        caption = f"Preprocessed Image with Detections ({processed_width} × {processed_height})"
                                    else:
                                        caption = f"Preprocessed Image - No Detections ({processed_width} × {processed_height})"
                                    st.image(preprocessed_img, caption=caption, width='stretch')
                            else:
                                st.warning(f"No image data available for {filename}")
                            
                            # Display inference information in 1x4 layout
                            st.markdown("---")
                            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                            
                            with info_col1:
                                # Display model info
                                if isinstance(result, dict):
                                    model = result.get("model")
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
                                if isinstance(result, dict):
                                    preprocess = result.get("preprocess")
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
                                if isinstance(result, dict):
                                    postprocess = result.get("postprocess")
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
                                if isinstance(result, dict):
                                    perform_time = result.get("perform_time")
                                    if perform_time:
                                        st.metric("Processing Time", f"{perform_time:.3f}s")
                                    else:
                                        st.metric("Processing Time", "N/A")
                                    
                                    # Display error if present
                                    if 'error' in result:
                                        st.error(f"**Error:** {result.get('error')}")
                                        if 'message' in result:
                                            st.caption(f"Message: {result.get('message')}")
                                        st.caption("⚠️ Inference NOT saved")
                                    else:
                                        challenge_id = result.get("challenge_id")
                                        if challenge_id:
                                            st.caption(f"✓ Saved (challenge_id: {challenge_id})")

                            # Display results table using standard schema
                            if detections and len(detections) > 0:
                                # Filter for valid detections (must have bbox with 4 elements)
                                valid_detections_for_table = [
                                    d for d in detections 
                                    if isinstance(d, dict) 
                                    and 'bbox' in d 
                                    and isinstance(d.get('bbox'), (list, tuple))
                                    and len(d.get('bbox', [])) >= 4
                                    and all(isinstance(x, (int, float)) for x in d.get('bbox', [])[:4])
                                ]

                                if valid_detections_for_table and len(valid_detections_for_table) > 0:
                                    st.markdown(f"**Detection Results for {filename}:**")
                                    table_data = []
                                    for idx, detection in enumerate(valid_detections_for_table):
                                        bbox = detection.get('bbox', [])
                                        bbox_str = f"[{', '.join([str(int(x)) for x in bbox[:4]])}]" if len(bbox) >= 4 else "N/A"
                                        table_data.append({
                                            "ID": idx + 1,
                                            "Class": detection.get('class', 'Unknown'),
                                            "Confidence": f"{detection.get('confidence', 0.0):.4f}",
                                            "Bounding Box": bbox_str,
                                            "Coordinates": f"({bbox[0]:.1f}, {bbox[1]:.1f}) to ({bbox[2]:.1f}, {bbox[3]:.1f})" if len(bbox) >= 4 else "N/A"
                                        })
                                    if table_data:
                                        st.dataframe(table_data, width='stretch')
                            else:
                                # Show clear message when no detections found
                                if detections and len(detections) == 0:
                                    st.info(f"**No detections found for {filename}**")
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
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Images Processed", len(accepted))
                    with col2:
                        if isinstance(first_result, dict) and 'perform_time' in first_result:
                            st.metric("Processing Time", f"{first_result.get('perform_time', 0):.3f}s")

                    # Display model, preprocessing, postprocessing, and performance info
                    if isinstance(first_result, dict):
                        model = first_result.get('model')
                        if model:
                            model_id = model.get('model_id', 'N/A')
                            model_name = model.get('model_name', 'N/A')
                            st.info(f"**Model ID:** {model_id}")
                            st.caption(f"Model Name: {model_name}")
                        else:
                            st.info("**Model ID:** N/A")
                            st.caption("Model Name: N/A")
                        
                        st.markdown("---")
                        
                        # Display preprocessing applied
                        preprocess = first_result.get('preprocess')
                        if preprocess:
                            preprocess_id = preprocess.get('preprocess_id', 'N/A')
                            applied_steps = preprocess.get('applied_steps', [])
                            st.info(f"**Preprocessing Applied:** {preprocess_id}")
                            if applied_steps and isinstance(applied_steps, list) and len(applied_steps) > 0:
                                step_names = [step.get('operation', '') for step in applied_steps if isinstance(step, dict)]
                                if step_names:
                                    st.caption(f"Steps: {', '.join(step_names)}")
                                else:
                                    st.caption("No preprocessing steps applied")
                            else:
                                st.caption("No preprocessing steps applied")
                        else:
                            st.info("**Preprocessing Applied:** N/A")
                            st.caption("No preprocessing applied")
                        
                        st.markdown("---")
                        
                        # Display postprocessing applied
                        postprocess = first_result.get('postprocess')
                        if postprocess:
                            postprocess_name = postprocess.get('name', 'N/A')
                            postprocess_id = postprocess.get('postprocess_id', 'N/A')
                            st.info(f"**Postprocessing Applied:** {postprocess_id}")
                            if postprocess_name and postprocess_name != 'N/A':
                                st.caption(f"Postprocess Name: {postprocess_name}")
                            
                            steps = postprocess.get('steps', [])
                            if isinstance(steps, list) and len(steps) > 0:
                                operations = [step.get('operation', 'Unknown') for step in steps if isinstance(step, dict)]
                                if operations:
                                    st.caption(f"Operations: {', '.join(operations)}")
                                else:
                                    st.caption("No postprocessing operations")
                            else:
                                st.caption("No postprocessing operations")
                        else:
                            st.info("**Postprocessing Applied:** N/A")
                            st.caption("No postprocessing applied")
                        
                        st.markdown("---")
                        
                        # Display performance
                        perform_time = first_result.get('perform_time')
                        if perform_time:
                            st.metric("Processing Time", f"{perform_time:.3f}s")
                        else:
                            st.metric("Processing Time", "N/A")
                        
                        st.markdown("---")

                    # Display each image with results
                    # For batch results, the 'results' field contains an array of batch items
                    # Each batch item has: {image_index, results: [detections]} per standard schema
                    batch_results = first_result.get('results', []) if isinstance(first_result, dict) else []

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
                            # Check if first item is a single image result (not batch format)
                            elif isinstance(first_item_result, dict):
                                first_results = first_item_result.get('results', [])
                                # If results is a direct list of detections (not batch format), it's a sample
                                if isinstance(first_results, list) and len(first_results) > 0:
                                    if not isinstance(first_results[0], dict) or 'image_index' not in first_results[0]:
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
                            # Try to get result by image_index first
                            # Note: image_index in API is 1-based (1, 2, 3, ...)
                            image_result_data = batch_results_by_index.get(image_index)

                            # Fallback: try by batch index if not found by image_index
                            if image_result_data is None and batch_idx >= 0 and batch_idx < len(batch_results):
                                image_result_data = batch_results[batch_idx]

                        st.markdown(f"#### {filename}")

                        # Extract detections using standard schema
                        detections_to_draw = []
                        if has_sample_tile and img_idx == 0:
                            # Single image result
                            detections_to_draw = extract_detections(result) if isinstance(result, dict) else []
                        elif image_result_data:
                            # Batch item result - extract detections from the batch item
                            detections_to_draw = image_result_data.get('results', [])
                            if not isinstance(detections_to_draw, list):
                                detections_to_draw = []
                        
                        # Display images in 1x2 layout (original and preprocessed)
                        original_img = None
                        preprocessed_img = None
                        
                        # Get original image from data_url
                        if data_url:
                            try:
                                b64_part = data_url.split(",", 1)[1] if "," in data_url else data_url
                                original_img_bytes = base64.b64decode(b64_part)
                                original_img = Image.open(io.BytesIO(original_img_bytes))
                            except Exception as e:
                                st.warning(f"Failed to load original image: {e}")
                        
                        # Get preprocessed image from API response (use first_result for batch)
                        processed_image_b64 = None
                        if has_sample_tile and img_idx == 0:
                            processed_image_b64 = result.get("processed_image") if isinstance(result, dict) else None
                        else:
                            # For batch items, use the first_result's processed_image if available
                            processed_image_b64 = first_result.get("processed_image") if isinstance(first_result, dict) else None
                        
                        if processed_image_b64:
                            try:
                                processed_img_bytes = base64.b64decode(processed_image_b64)
                                preprocessed_img = Image.open(io.BytesIO(processed_img_bytes))
                            except Exception as e:
                                st.warning(f"Failed to decode preprocessed image: {e}")
                                # Fallback to original if available
                                if original_img:
                                    preprocessed_img = original_img.copy()
                        elif original_img:
                            # No preprocessing was applied, use original
                            preprocessed_img = original_img.copy()
                        
                        # Draw bounding boxes on preprocessed image if there are detections
                        if preprocessed_img and detections_to_draw and len(detections_to_draw) > 0:
                            # Filter for valid detections (must have bbox with 4 elements)
                            valid_detections = [
                                d for d in detections_to_draw 
                                if isinstance(d, dict) 
                                and 'bbox' in d 
                                and isinstance(d.get('bbox'), (list, tuple))
                                and len(d.get('bbox', [])) >= 4
                            ]
                            if valid_detections:
                                draw = ImageDraw.Draw(preprocessed_img)
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
                        
                        # Display images in 1x2 layout
                        if original_img and preprocessed_img:
                            img_col1, img_col2 = st.columns(2)
                            
                            original_width, original_height = original_img.size
                            processed_width, processed_height = preprocessed_img.size
                            
                            with img_col1:
                                st.image(original_img, caption=f"Original Image ({original_width} × {original_height})", width='stretch')
                            
                            with img_col2:
                                if detections_to_draw and len(detections_to_draw) > 0:
                                    caption = f"Preprocessed Image with Detections ({processed_width} × {processed_height})"
                                else:
                                    caption = f"Preprocessed Image - No Detections ({processed_width} × {processed_height})"
                                st.image(preprocessed_img, caption=caption, width='stretch')
                        else:
                            st.warning(f"No image data available for {filename}")
                        
                        # Display inference information in 1x4 layout
                        st.markdown("---")
                        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                        
                        # Use first_result for batch metadata (all images share the same preprocessing/postprocessing)
                        display_result = first_result if isinstance(first_result, dict) else result
                        
                        with info_col1:
                            # Display model info
                            if isinstance(display_result, dict):
                                model = display_result.get("model")
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
                            if isinstance(display_result, dict):
                                preprocess = display_result.get("preprocess")
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
                            if isinstance(display_result, dict):
                                postprocess = display_result.get("postprocess")
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
                            if isinstance(display_result, dict):
                                perform_time = display_result.get("perform_time")
                                if perform_time:
                                    st.metric("Processing Time", f"{perform_time:.3f}s")
                                else:
                                    st.metric("Processing Time", "N/A")
                        
                        # Display results table using standard schema
                        table_display_result = None
                        if has_sample_tile and img_idx == 0:
                            # For sample tile, use the result directly (single image API response)
                            table_display_result = result
                        elif image_result_data:
                            # For batch tiles, use the individual image result from batch
                            table_display_result = image_result_data

                        # Extract detections using standard schema
                        detections_for_table = []
                        if table_display_result:
                            # For batch items: table_display_result is {image_index, results: [detections]}
                            # For single image: table_display_result is {results: [detections], ...}
                            detections_for_table = extract_detections(table_display_result) if isinstance(table_display_result, dict) else []

                        if detections_for_table and len(detections_for_table) > 0:
                            # Filter for valid detections (must have bbox with 4 elements)
                            valid_detections = [
                                d for d in detections_for_table 
                                if isinstance(d, dict) 
                                and 'bbox' in d 
                                and isinstance(d.get('bbox'), (list, tuple))
                                and len(d.get('bbox', [])) >= 4
                                and all(isinstance(x, (int, float)) for x in d.get('bbox', [])[:4])
                            ]

                            if valid_detections:
                                st.markdown(f"**Detection Results for {filename}:**")
                                table_data = []
                                for idx, detection in enumerate(valid_detections):
                                    bbox = detection.get('bbox', [])
                                    bbox_str = f"[{', '.join([str(int(x)) for x in bbox[:4]])}]" if len(bbox) >= 4 else "N/A"
                                    table_data.append({
                                        "ID": idx + 1,
                                        "Class": detection.get('class', 'Unknown'),
                                        "Confidence": f"{detection.get('confidence', 0.0):.4f}",
                                        "Bounding Box": bbox_str,
                                        "Coordinates": f"({bbox[0]:.1f}, {bbox[1]:.1f}) to ({bbox[2]:.1f}, {bbox[3]:.1f})" if len(bbox) >= 4 else "N/A"
                                    })
                                if table_data:
                                    st.dataframe(table_data, width='stretch')
                            else:
                                st.info(f"No valid detections for {filename} ({len(detections_for_table)} invalid result(s))")
                        else:
                            st.info(f"No detections found for {filename}")

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

                        # Extract detections using standard schema
                        detections = extract_detections(result) if isinstance(result, dict) else []
                        
                        # Display images in 1x2 layout (original and preprocessed)
                        original_img = None
                        preprocessed_img = None
                        
                        # Get original image from data_url
                        if data_url:
                            try:
                                b64_part = data_url.split(",", 1)[1] if "," in data_url else data_url
                                original_img_bytes = base64.b64decode(b64_part)
                                original_img = Image.open(io.BytesIO(original_img_bytes))
                            except Exception as e:
                                st.warning(f"Failed to load original image: {e}")
                        
                        # Get preprocessed image from API response
                        processed_image_b64 = result.get("processed_image") if isinstance(result, dict) else None
                        if processed_image_b64:
                            try:
                                processed_img_bytes = base64.b64decode(processed_image_b64)
                                preprocessed_img = Image.open(io.BytesIO(processed_img_bytes))
                            except Exception as e:
                                st.warning(f"Failed to decode preprocessed image: {e}")
                                # Fallback to original if available
                                if original_img:
                                    preprocessed_img = original_img.copy()
                        elif original_img:
                            # No preprocessing was applied, use original
                            preprocessed_img = original_img.copy()
                        
                        # Draw bounding boxes on preprocessed image if there are detections
                        if preprocessed_img and detections and len(detections) > 0:
                            # Filter for valid detections (must have bbox with 4 elements)
                            valid_detections = [
                                d for d in detections 
                                if isinstance(d, dict) 
                                and 'bbox' in d 
                                and isinstance(d.get('bbox'), (list, tuple))
                                and len(d.get('bbox', [])) >= 4
                                and all(isinstance(x, (int, float)) for x in d.get('bbox', [])[:4])
                            ]

                            if valid_detections:
                                draw = ImageDraw.Draw(preprocessed_img)
                                try:
                                    font = ImageFont.truetype("arial.ttf", 16)
                                except:
                                    try:
                                        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                                    except:
                                        font = ImageFont.load_default()

                                for detection in valid_detections:
                                    bbox = detection.get('bbox', [])
                                    if len(bbox) >= 4:
                                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                                        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                                        class_name = detection.get('class', 'Unknown')
                                        confidence = detection.get('confidence', 0.0)
                                        label = f"{class_name}: {confidence:.2f}"
                                        bbox_text = draw.textbbox((0, 0), label, font=font)
                                        text_width = bbox_text[2] - bbox_text[0]
                                        text_height = bbox_text[3] - bbox_text[1]
                                        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill="red")
                                        draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=font)
                        
                        # Display images in 1x2 layout
                        if original_img and preprocessed_img:
                            img_col1, img_col2 = st.columns(2)
                            
                            original_width, original_height = original_img.size
                            processed_width, processed_height = preprocessed_img.size
                            
                            with img_col1:
                                st.image(original_img, caption=f"Original Image ({original_width} × {original_height})", width='stretch')
                            
                            with img_col2:
                                if detections and len(detections) > 0:
                                    caption = f"Preprocessed Image with Detections ({processed_width} × {processed_height})"
                                else:
                                    caption = f"Preprocessed Image - No Detections ({processed_width} × {processed_height})"
                                st.image(preprocessed_img, caption=caption, width='stretch')
                        else:
                            st.warning(f"No image data available for {filename}")
                        
                        # Display inference information in 1x4 layout
                        st.markdown("---")
                        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                        
                        with info_col1:
                            # Display model info
                            if isinstance(result, dict):
                                model = result.get("model")
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
                            if isinstance(result, dict):
                                preprocess = result.get("preprocess")
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
                            if isinstance(result, dict):
                                postprocess = result.get("postprocess")
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
                            if isinstance(result, dict):
                                perform_time = result.get("perform_time")
                                if perform_time:
                                    st.metric("Processing Time", f"{perform_time:.3f}s")
                                else:
                                    st.metric("Processing Time", "N/A")
                                
                                # Display error if present
                                if 'error' in result:
                                    st.error(f"**Error:** {result.get('error')}")
                                    if 'message' in result:
                                        st.caption(f"Message: {result.get('message')}")
                                    st.caption("⚠️ Inference NOT saved")
                                else:
                                    challenge_id = result.get("challenge_id")
                                    if challenge_id:
                                        st.caption(f"✓ Saved (challenge_id: {challenge_id})")

                        # Display results table using standard schema
                        if detections and len(detections) > 0:
                            # Filter for valid detections (must have bbox with 4 elements)
                            valid_detections_for_table = [
                                d for d in detections 
                                if isinstance(d, dict) 
                                and 'bbox' in d 
                                and isinstance(d.get('bbox'), (list, tuple))
                                and len(d.get('bbox', [])) >= 4
                                and all(isinstance(x, (int, float)) for x in d.get('bbox', [])[:4])
                            ]

                            if valid_detections_for_table and len(valid_detections_for_table) > 0:
                                st.markdown("### Detection Results")
                                table_data = []
                                for idx, detection in enumerate(valid_detections_for_table):
                                    bbox = detection.get('bbox', [])
                                    bbox_str = f"[{', '.join([str(int(x)) for x in bbox[:4]])}]" if len(bbox) >= 4 else "N/A"
                                    table_data.append({
                                        "ID": idx + 1,
                                        "Class": detection.get('class', 'Unknown'),
                                        "Confidence": f"{detection.get('confidence', 0.0):.4f}",
                                        "Bounding Box": bbox_str,
                                        "Coordinates": f"({bbox[0]:.1f}, {bbox[1]:.1f}) to ({bbox[2]:.1f}, {bbox[3]:.1f})" if len(bbox) >= 4 else "N/A"
                                    })
                                if table_data:
                                    st.dataframe(table_data, width='stretch')
                            else:
                                st.info("**No valid detections found**")
                        else:
                            st.info("**No detections found**")

                        with st.expander("View Full Response"):
                            st.json(result)

                        if idx < len(accepted) - 1:
                            st.markdown("---")
        else:
            st.warning("Crawl finished. No matched challenge type found for the challenge.")
        st.info(f"Elapsed: {elapsed:.2f}s")
        progress.progress(100)
