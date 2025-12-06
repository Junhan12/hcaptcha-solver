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
    run_crawl_once,
    decompress_image_to_base64,
)

def render(progress, status):
    """Render the Auto Crawler, Solver, and Clicker page."""
    # Use existing "Crawl -> API" functionality
    if st.button("Crawl Demo Site And Solve Challenges", key="crawl_demo_button"):
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

                                # Always show processed image if available, otherwise fallback to data_url
                                display_img_b64 = None
                                if processed_img_b64:
                                    display_img_b64 = processed_img_b64
                                elif data_url:
                                    # Extract base64 from data_url if no processed_image
                                    try:
                                        b64_part = data_url.split(",", 1)[1] if "," in data_url else data_url
                                        display_img_b64 = b64_part
                                    except:
                                        pass

                                if display_img_b64:
                                    try:
                                        img_bytes = base64.b64decode(display_img_b64)
                                        img = Image.open(io.BytesIO(img_bytes))

                                        # Get processed image dimensions
                                        processed_width, processed_height = img.size

                                        # Draw bounding boxes if there are valid detections
                                        valid_detections = []
                                        if results and isinstance(results, list) and len(results) > 0:
                                            # Filter for valid detections (must have bbox with 4 elements)
                                            valid_detections = [
                                                r for r in results 
                                                if isinstance(r, dict) 
                                                and 'error' not in r 
                                                and 'bbox' in r 
                                                and isinstance(r.get('bbox'), (list, tuple))
                                                and len(r.get('bbox', [])) >= 4
                                                and all(isinstance(x, (int, float)) for x in r.get('bbox', [])[:4])
                                            ]

                                            if valid_detections:
                                                draw = ImageDraw.Draw(img)
                                                try:
                                                    font = ImageFont.truetype("arial.ttf", 16)
                                                except:
                                                    try:
                                                        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                                                    except:
                                                        font = ImageFont.load_default()

                                                for result_item in valid_detections:
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

                                        # Always display the image (with or without detections) at processed size
                                        if valid_detections:
                                            caption = f"{filename} (Challenge {challenge_idx}) - With Detections ({processed_width} × {processed_height})"
                                        else:
                                            caption = f"{filename} (Challenge {challenge_idx}) - No Detections ({processed_width} × {processed_height})"

                                        st.image(img, caption=caption, width='stretch')
                                    except Exception as e:
                                        st.error(f"Failed to display image: {e}")
                                else:
                                    # Fallback: try to display from data_url if no processed_image
                                    if data_url:
                                        try:
                                            b64_part = data_url.split(",", 1)[1] if "," in data_url else data_url
                                            img_bytes = base64.b64decode(b64_part)
                                            fallback_img = Image.open(io.BytesIO(img_bytes))
                                            fallback_width, fallback_height = fallback_img.size
                                            st.image(fallback_img, caption=f"{filename} (Challenge {challenge_idx}) - Original Image ({fallback_width} × {fallback_height})", width='stretch')
                                        except Exception:
                                            st.write(f"{filename} (Challenge {challenge_idx})")
                                    else:
                                        st.warning(f"No image data available for {filename}")

                            with col2:
                                if isinstance(result, dict):
                                    preprocess = result.get("preprocess")
                                    if preprocess:
                                        st.info(f"**Preprocessing:** {preprocess.get('preprocess_id', 'N/A')}")

                                    postprocess = result.get("postprocess")
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

                                    if 'perform_time' in result:
                                        st.metric("Processing Time", f"{result.get('perform_time', 0):.3f}s")

                                    if 'challenge_id' in result:
                                        st.caption(f"**Challenge ID:** {result.get('challenge_id')}")

                                    if 'error' in result:
                                        st.error(f"**Error:** {result.get('error')}")
                                        st.caption("X Inference NOT saved (errors are not stored)")
                                    elif results:
                                        if isinstance(results, list):
                                            # Filter for valid detections
                                            valid_detections = [
                                                r for r in results 
                                                if isinstance(r, dict) 
                                                and 'error' not in r 
                                                and 'bbox' in r 
                                                and isinstance(r.get('bbox'), (list, tuple))
                                                and len(r.get('bbox', [])) >= 4
                                                and all(isinstance(x, (int, float)) for x in r.get('bbox', [])[:4])
                                            ]

                                            if len(valid_detections) > 0:
                                                st.success(f"**Detections:** {len(valid_detections)} objects")
                                                st.caption("/ Inference saved to database")
                                            else:
                                                if len(results) > 0:
                                                    st.info(f"**No valid detections found** ({len(results)} invalid result(s))")
                                                else:
                                                    st.info("**No detections found**")
                                                st.caption("/ Inference saved to database")
                                        else:
                                            st.info("**No detections found**")
                                            st.caption("/ Inference saved to database")
                                    else:
                                        st.info("**No detections found**")
                                        st.caption("/ Inference saved to database")

                            # Display results table only if there are valid detections
                            valid_detections_for_table = []
                            if results and isinstance(results, list) and len(results) > 0:
                                valid_detections_for_table = [
                                    r for r in results 
                                    if isinstance(r, dict) 
                                    and 'error' not in r 
                                    and 'bbox' in r 
                                    and isinstance(r.get('bbox'), (list, tuple))
                                    and len(r.get('bbox', [])) >= 4
                                    and all(isinstance(x, (int, float)) for x in r.get('bbox', [])[:4])
                                ]

                            if valid_detections_for_table and len(valid_detections_for_table) > 0:
                                st.markdown(f"**Detection Results for {filename}:**")
                                table_data = []
                                for result_idx, result_item in enumerate(valid_detections_for_table):
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
                                    st.dataframe(table_data, width='stretch')
                            else:
                                # Show clear message when no detections found
                                if results and isinstance(results, list) and len(results) > 0:
                                    st.info(f"**No valid detections found for {filename}** ({len(results)} invalid result(s))")
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

                    # Display each image with results
                    # For batch results, the 'results' field contains an array of batch items
                    # Each batch item has: {image_index, image_name, results: [detections]}
                    batch_results = first_result.get('results', []) if isinstance(first_result, dict) else []
                    processed_images = first_result.get('processed_images', []) if isinstance(first_result, dict) else []

                    # Debug: Log batch results structure
                    print(f"Debug: batch_results type={type(batch_results)}, length={len(batch_results) if isinstance(batch_results, list) else 'N/A'}")
                    if batch_results and len(batch_results) > 0:
                        print(f"Debug: First batch item keys: {list(batch_results[0].keys()) if isinstance(batch_results[0], dict) else 'N/A'}")
                        if isinstance(batch_results[0], dict):
                            print(f"Debug: First batch item image_index: {batch_results[0].get('image_index')}")
                            print(f"Debug: First batch item results type: {type(batch_results[0].get('results'))}, length={len(batch_results[0].get('results', [])) if isinstance(batch_results[0].get('results'), list) else 'N/A'}")

                    # Create a mapping from image_index to batch result for easier lookup
                    batch_results_by_index = {}
                    for batch_item in batch_results:
                        if isinstance(batch_item, dict) and 'image_index' in batch_item:
                            batch_results_by_index[batch_item['image_index']] = batch_item

                    print(f"Debug: batch_results_by_index keys: {list(batch_results_by_index.keys())}")

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
                            # Try to get result by image_index first
                            # Note: image_index in API is 1-based (1, 2, 3, ...)
                            image_result_data = batch_results_by_index.get(image_index)

                            # Fallback: try by batch index if not found by image_index
                            if image_result_data is None and batch_idx >= 0 and batch_idx < len(batch_results):
                                image_result_data = batch_results[batch_idx]
                                print(f"Debug: Using fallback - found result by batch_idx={batch_idx} for image_index={image_index}")

                            # Debug: Log result retrieval
                            if image_result_data:
                                img_res = image_result_data.get('results', [])
                                print(f"Debug: Found image_result_data for {filename}: image_index={image_result_data.get('image_index')}, results type={type(img_res)}, results length={len(img_res) if isinstance(img_res, list) else 'N/A'}")
                                if isinstance(img_res, list) and len(img_res) > 0:
                                    print(f"Debug: First result keys: {list(img_res[0].keys()) if isinstance(img_res[0], dict) else 'N/A'}")
                                    if isinstance(img_res[0], dict) and 'bbox' in img_res[0]:
                                        print(f"Debug: First result bbox: {img_res[0].get('bbox')}, bbox type: {type(img_res[0].get('bbox'))}, bbox length: {len(img_res[0].get('bbox', [])) if isinstance(img_res[0].get('bbox'), (list, tuple)) else 'N/A'}")
                            else:
                                print(f"Debug: Could not find result for img_idx={img_idx}, batch_idx={batch_idx}, image_index={image_index}")
                                print(f"Debug: batch_results_by_index keys: {list(batch_results_by_index.keys())}")
                                print(f"Debug: batch_results length: {len(batch_results)}")
                                if batch_results and len(batch_results) > 0:
                                    print(f"Debug: First batch result keys: {list(batch_results[0].keys()) if isinstance(batch_results[0], dict) else 'N/A'}")
                                    if isinstance(batch_results[0], dict):
                                        print(f"Debug: First batch result image_index: {batch_results[0].get('image_index')}")

                        st.markdown(f"#### {filename}")

                        col1, col2 = st.columns([2, 1])

                        with col1:
                            display_img_b64 = None
                            if has_sample_tile and img_idx == 0:
                                # For sample tile, prefer processed_image, fallback to data_url
                                if isinstance(result, dict) and result.get('processed_image'):
                                    display_img_b64 = result.get('processed_image')
                                elif data_url:
                                    # Extract base64 from data_url if no processed_image
                                    try:
                                        b64_part = data_url.split(",", 1)[1] if "," in data_url else data_url
                                        display_img_b64 = b64_part
                                    except:
                                        pass
                            elif processed_images:
                                batch_img_idx = img_idx - sample_offset
                                if batch_img_idx >= 0 and batch_img_idx < len(processed_images):
                                    display_img_b64 = processed_images[batch_img_idx]

                            if display_img_b64:
                                try:
                                    img_bytes = base64.b64decode(display_img_b64)
                                    img = Image.open(io.BytesIO(img_bytes))

                                    # Get processed image dimensions
                                    processed_width, processed_height = img.size

                                    results_to_draw = None
                                    if has_sample_tile and img_idx == 0:
                                        results_to_draw = result.get('results', []) if isinstance(result, dict) else []
                                    elif image_result_data:
                                        results_to_draw = image_result_data.get('results', [])

                                    valid_detections = []
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

                                    # Always display the image (with or without detections) at processed size
                                    if valid_detections:
                                        caption = f"{filename} - With Detections ({processed_width} × {processed_height})"
                                    else:
                                        caption = f"{filename} - No Detections ({processed_width} × {processed_height})"

                                    st.image(img, caption=caption, width='stretch')
                                except Exception as e:
                                    st.error(f"Failed to display image: {e}")
                            else:
                                if data_url:
                                    try:
                                        b64_part = data_url.split(",", 1)[1] if "," in data_url else data_url
                                        img_bytes = base64.b64decode(b64_part)
                                        fallback_img = Image.open(io.BytesIO(img_bytes))
                                        fallback_width, fallback_height = fallback_img.size
                                        st.image(fallback_img, caption=f"{filename} - Original Image ({fallback_width} × {fallback_height})", width='stretch')
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
                                # For sample tile, use the result directly (single image API response)
                                table_display_result = result
                            elif image_result_data:
                                # For batch tiles, use the individual image result from batch
                                table_display_result = image_result_data

                            if table_display_result:
                                # Extract results from the appropriate structure
                                # For batch items: table_display_result is {image_index, image_name, results: [detections]}
                                # For single image: table_display_result is {results: [detections], ...}
                                img_results = table_display_result.get('results', [])

                                # Debug: Log the results structure
                                print(f"Debug: table_display_result for {filename}: type={type(img_results)}, length={len(img_results) if isinstance(img_results, list) else 'N/A'}")
                                if isinstance(img_results, list) and len(img_results) > 0:
                                    print(f"Debug: First result item type: {type(img_results[0])}, keys: {list(img_results[0].keys()) if isinstance(img_results[0], dict) else 'N/A'}")

                                if isinstance(img_results, list):
                                    if len(img_results) > 0:
                                        # Filter for valid detections (must have bbox with 4 elements)
                                        # Only include detections that have a valid bbox with at least 4 elements
                                        valid_detections = [
                                            r for r in img_results 
                                            if isinstance(r, dict) 
                                            and 'error' not in r 
                                            and 'bbox' in r 
                                            and isinstance(r.get('bbox'), (list, tuple))
                                            and len(r.get('bbox', [])) >= 4
                                            and all(isinstance(x, (int, float)) for x in r.get('bbox', [])[:4])
                                        ]

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
                                                st.dataframe(table_data, width='stretch')
                                        else:
                                            # No valid detections - check if there are any results at all
                                            if len(img_results) > 0:
                                                # There are results but they're invalid - show message
                                                st.info(f"No valid detections for {filename} ({len(img_results)} invalid result(s))")
                                            else:
                                                # Empty results - no detections
                                                st.info(f"No detections found for {filename}")
                                    else:
                                        # Empty list - no detections
                                        st.info(f"No detections found for {filename}")
                                elif isinstance(img_results, dict):
                                    if 'error' in img_results:
                                        st.error(f"Error: {img_results.get('error')}")
                                    elif 'message' in img_results:
                                        st.info(f"Message: {img_results.get('message')}")
                                    else:
                                        st.warning(f"Unexpected result format for {filename}")
                                else:
                                    st.info(f"No detections found for {filename} (results type: {type(img_results)})")
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

                            # Always show processed image if available, otherwise fallback to data_url
                            display_img_b64 = None
                            if processed_img_b64:
                                display_img_b64 = processed_img_b64
                            elif data_url:
                                # Extract base64 from data_url if no processed_image
                                try:
                                    b64_part = data_url.split(",", 1)[1] if "," in data_url else data_url
                                    display_img_b64 = b64_part
                                except:
                                    pass

                            if display_img_b64:
                                try:
                                    img_bytes = base64.b64decode(display_img_b64)
                                    img = Image.open(io.BytesIO(img_bytes))

                                    # Get processed image dimensions
                                    processed_width, processed_height = img.size

                                    # Draw bounding boxes if there are valid detections
                                    valid_detections = []
                                    if results and isinstance(results, list) and len(results) > 0:
                                        # Filter for valid detections (must have bbox with 4 elements)
                                        valid_detections = [
                                            r for r in results 
                                            if isinstance(r, dict) 
                                            and 'error' not in r 
                                            and 'bbox' in r 
                                            and isinstance(r.get('bbox'), (list, tuple))
                                            and len(r.get('bbox', [])) >= 4
                                            and all(isinstance(x, (int, float)) for x in r.get('bbox', [])[:4])
                                        ]

                                        if valid_detections:
                                            draw = ImageDraw.Draw(img)
                                            try:
                                                font = ImageFont.truetype("arial.ttf", 16)
                                            except:
                                                try:
                                                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                                                except:
                                                    font = ImageFont.load_default()

                                            for result_item in valid_detections:
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

                                    # Always display the image (with or without detections) at processed size
                                    if valid_detections:
                                        caption = f"Processed Image with Detections ({processed_width} × {processed_height})"
                                    else:
                                        caption = f"Processed Image (No Detections) ({processed_width} × {processed_height})"

                                    st.image(img, caption=caption, width='stretch')
                                except Exception as e:
                                    st.error(f"Failed to display image: {e}")
                            else:
                                if data_url:
                                    try:
                                        b64_part = data_url.split(",", 1)[1] if "," in data_url else data_url
                                        img_bytes = base64.b64decode(b64_part)
                                        fallback_img = Image.open(io.BytesIO(img_bytes))
                                        fallback_width, fallback_height = fallback_img.size
                                        st.image(fallback_img, caption=f"{filename} - Original Image ({fallback_width} × {fallback_height})", width='stretch')
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

                                if 'perform_time' in result:
                                    st.metric("Processing Time", f"{result.get('perform_time', 0):.3f}s")

                                if 'challenge_id' in result:
                                    st.caption(f"**Challenge ID:** {result.get('challenge_id')}")

                                if 'error' in result:
                                    st.error(f"**Error:** {result.get('error')}")
                                    st.caption("X Inference NOT saved (errors are not stored)")
                                elif results:
                                    if isinstance(results, list):
                                        # Filter for valid detections
                                        valid_detections = [
                                            r for r in results 
                                            if isinstance(r, dict) 
                                            and 'error' not in r 
                                            and 'bbox' in r 
                                            and isinstance(r.get('bbox'), (list, tuple))
                                            and len(r.get('bbox', [])) >= 4
                                            and all(isinstance(x, (int, float)) for x in r.get('bbox', [])[:4])
                                        ]

                                        if len(valid_detections) > 0:
                                            st.success(f"**Detections:** {len(valid_detections)} objects")
                                            st.caption("/ Inference saved to database")
                                        else:
                                            if len(results) > 0:
                                                st.info(f"**No valid detections found** ({len(results)} invalid result(s))")
                                            else:
                                                st.info("**No detections found**")
                                            st.caption("/ Inference saved to database")
                                    elif isinstance(results, dict) and 'message' in results:
                                        st.info(f"**Message:** {results.get('message')}")
                                        st.caption("/ Inference saved to database")
                                    else:
                                        st.info("**No detections found**")
                                        st.caption("/ Inference saved to database")
                                else:
                                    st.info("**No detections found**")
                                    st.caption("/ Inference saved to database")

                        # Display results table only if there are valid detections
                        valid_detections_for_table = []
                        if results and isinstance(results, list) and len(results) > 0:
                            valid_detections_for_table = [
                                r for r in results 
                                if isinstance(r, dict) 
                                and 'error' not in r 
                                and 'bbox' in r 
                                and isinstance(r.get('bbox'), (list, tuple))
                                and len(r.get('bbox', [])) >= 4
                                and all(isinstance(x, (int, float)) for x in r.get('bbox', [])[:4])
                            ]

                        if valid_detections_for_table and len(valid_detections_for_table) > 0:
                            st.markdown("### Detection Results")
                            table_data = []
                            for result_idx, result_item in enumerate(valid_detections_for_table):
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
                                st.dataframe(table_data, width='stretch')
                        else:
                            # Show clear message when no detections found
                            if results and isinstance(results, list) and len(results) > 0:
                                st.info(f"**No valid detections found** ({len(results)} invalid result(s))")
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
