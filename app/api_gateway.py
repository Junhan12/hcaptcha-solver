import threading
import tempfile
import os
import shutil
from flask import Flask, request, jsonify
import base64
import zlib
from bson.binary import Binary

# Ensure FLASK_HOST/FLASK_PORT are available regardless of how this module is executed
try:
    from .config import FLASK_HOST, FLASK_PORT
except ImportError:
    from app.config import FLASK_HOST, FLASK_PORT

try:
    # Prefer package-relative imports when run as a module: `python -m app.api_gateway`
    from .solver import solve_captcha
    from .database import get_model_config
    from .database import create_challenge, get_model_for_challenge
    from .database import upsert_model, list_models
    from .database import get_preprocess_for_model, get_postprocess_for_model
    from .database import create_inference
    from .database import create_activity_log
    from .preprocess import apply_preprocess
except Exception:
    # Fallback when executed in contexts that expect absolute imports with project root on sys.path
    from app.solver import solve_captcha
    from app.database import get_model_config
    from app.database import create_challenge, get_model_for_challenge
    from app.database import upsert_model, list_models
    from app.database import get_preprocess_for_model, get_postprocess_for_model
    from app.database import create_inference
    from app.database import create_activity_log
    from app.preprocess import apply_preprocess

from app.utils.logger import get_logger

log = get_logger("api")

app = Flask(__name__)

# =========================
# Headless API endpoints (for crawler, clicker, and Streamlit upload only)
# =========================

@app.route('/solve_hcaptcha', methods=['POST'])
def solve_hcaptcha():
    """
    Headless API endpoint for processing single image challenges.
    
    Contract (per hcaptcha-rules.mdc):
    - Request: image bytes (multipart/form-data 'image') + question string (form 'question')
    - Response: {
        'results': list of detection dicts (each with 'bbox', 'class', 'confidence'),
        'model': optional model info dict,
        'perform_time': optional float,
        'challenge_id': optional string,
        'message': optional string,
        'error': optional string
      }
    
    Detection dict format:
    - 'bbox': [x_min, y_min, x_max, y_max] in intrinsic image coordinates (model space)
    - 'class': predicted class label (str)
    - 'confidence': float between 0-1
    
    Intended usage:
    - Crawler (client/crawler.py): Sends canvas images from hCAPTCHA challenges
    - External clients via HTTP API
    """
    # Validate request
    if 'image' not in request.files:
        return jsonify({
            'error': 'missing_image',
            'message': 'No image file provided in request',
            'results': []
        }), 400
    
    img = request.files['image']
    question = request.form.get('question', '')
    
    # For now: create challenge, derive challenge_type, and only retrieve model
    import time
    t0 = time.time()
    # Read raw bytes for compression and model use
    img_bytes = img.read()
    # Rewind the stream for any downstream consumer that might need it
    try:
        img.stream.seek(0)
    except Exception:
        pass
    
    # Check if question matches any challenge_type before saving
    from .database import _find_challenge_type_for_question
    challenge_type_matched = False
    challenge_id = None
    
    # Find challenge_type and get model directly from it
    ct_doc = None
    model_doc = None
    challenge_type_matched = False
    challenge_id = None
    
    if question:
        ct_doc = _find_challenge_type_for_question(question)
        if ct_doc:
            challenge_type_matched = True
            challenge_type_id = ct_doc.get("challenge_type_id")
            log.info(f"Challenge type matched: {challenge_type_id}")
            
            # Get model directly from challenge_type document
            model_id = ct_doc.get("model_id")
            if model_id:
                from .database import get_model_by_id
                model_doc = get_model_by_id(model_id)
                if model_doc:
                    log.info(f"Model found: {model_id}")
                else:
                    log.warning(f"challenge_type {challenge_type_id} has model_id {model_id}, but model not found in database")
            else:
                log.warning(f"challenge_type {challenge_type_id} has no model_id set")
            
            # Only save challenge if it matches a challenge_type
            challenge_id = f"ch-{int(t0)}"
            try:
                payload_img = Binary(zlib.compress(img_bytes))
            except Exception:
                payload_img = img_bytes
            create_challenge(
                challenge_id=challenge_id,
                challenge_questions=question or "",
                challenge_img=payload_img,
                challenge_type_id=challenge_type_id,
            )
        else:
            # No matching challenge_type - don't save challenge
            log.info(f"No matching challenge_type for question: {question}")
            challenge_id = None
    else:
        # No question provided - don't save challenge
        log.info("No question provided - skipping challenge save")
    
    # Apply preprocessing if model has preprocess_id
    preprocess_meta = None
    processed_img_bytes = img_bytes
    if model_doc:
        try:
            profile = get_preprocess_for_model(model_doc)
            if profile:
                processed_img_bytes, applied_steps, _ = apply_preprocess(img_bytes, profile)
                preprocess_meta = {
                    'preprocess_id': profile.get('preprocess_id'),
                    'applied_steps': applied_steps,
                }
                # Use processed bytes for storage
                img_bytes = processed_img_bytes
        except Exception as e:
            log.warning(f"Preprocessing error: {e}")
            preprocess_meta = None
    
    # Retrieve postprocess profile for model inference
    postprocess_meta = None
    if model_doc:
        try:
            postprocess_profile = get_postprocess_for_model(model_doc)
            if postprocess_profile:
                postprocess_meta = {
                    'postprocess_id': postprocess_profile.get('postprocess_id'),
                    'name': postprocess_profile.get('name'),
                    'steps': postprocess_profile.get('steps', []),  # List of operations
                }
        except Exception as e:
            log.warning(f"Postprocess retrieval error: {e}")
            postprocess_meta = None
    
    # Determine message when no matching challenge_type
    message = None
    error = None
    if not challenge_type_matched:
        message = "no match challenge type found"
        error = "no match challenge type found"
        # If no match, return early without processing
        if not challenge_id:
            return jsonify({
                'error': error,
                'message': message,
                'results': [],
            })
    
    # Also compute config view for convenience
    config = get_model_config(question)
    
    # Run model inference with postprocess steps
    inference_results = []
    inference_start_time = time.time()
    try:
        if model_doc:
            # Pass full postprocess profile for modular postprocessing
            postprocess_profile = None
            if postprocess_meta:
                # Pass the full postprocess_profile structure
                postprocess_profile = {
                    'postprocess_id': postprocess_meta.get('postprocess_id'),
                    'name': postprocess_meta.get('name'),
                    'steps': postprocess_meta.get('steps', [])
                }
            
            # Use processed image for inference with postprocess profile
            inference_results = solve_captcha(
                processed_img_bytes, 
                question, 
                config,
                postprocess_profile=postprocess_profile
            )
            # Normalize inference_results to always be a list of detections
            # solve_captcha returns: list, dict with 'error', or dict with 'message'
            if isinstance(inference_results, dict):
                if 'error' in inference_results:
                    error = inference_results['error']
                    message = f"Inference failed: {error}"
                    inference_results = []
                elif 'message' in inference_results:
                    # No detections found - return empty list
                    message = inference_results.get('message', 'no detected output')
                    inference_results = []
                else:
                    # Unexpected dict format - treat as error
                    error = 'invalid_inference_result'
                    message = 'Inference returned unexpected format'
                    inference_results = []
            elif not isinstance(inference_results, list):
                # Invalid format - return empty list
                error = 'invalid_inference_result'
                message = 'Inference returned invalid format'
                inference_results = []
        else:
            # No model document found
            error = 'no model selected'
            message = 'No model available for this challenge type'
            inference_results = []
    except Exception as e:
        log.error(f"Inference error: {e}")
        error = 'inference_error'
        message = f"Inference failed: {str(e)}"
        inference_results = []
    
    inference_end_time = time.time()
    inference_speed = inference_end_time - inference_start_time
    t1 = time.time()
    
    # Save inference to inference collection only if no errors
    has_error = error is not None
    
    if not has_error and inference_results:
        try:
            inference_id = f"inf-{int(time.time() * 1000)}"  # Use milliseconds for uniqueness
            model_id = config.get('model_id') or (model_doc.get('model_id') if model_doc else None)
            
            # Ensure detection_results is in the correct format (list or dict)
            detection_results = inference_results
            if not isinstance(detection_results, (list, dict)):
                detection_results = []
            
            # Only save inference if challenge was saved (i.e., challenge_type matched)
            if model_id and challenge_id:
                created_inf = create_inference(
                    inference_id=inference_id,
                    inference_speed=inference_speed,
                    detection_results=detection_results,
                    model_id=model_id,
                    challenge_id=challenge_id
                )
                log.debug(f"Saved inference record: {inference_id}", indent=1)
                # Save activity log entry
                try:
                    log_id = f"act-{int(time.time() * 1000)}"
                    create_activity_log(
                        log_id=log_id,
                        challenge_id=challenge_id,
                        model_id=model_id,
                        inference_id=inference_id
                    )
                    log.debug(f"Saved activity_log record: {log_id}", indent=1)
                except Exception as e:
                    log.warning(f"Failed to save activity_log: {e}")
            else:
                if not challenge_id:
                    log.debug("Skipping inference save: challenge not saved (no matching challenge_type)", indent=1)
                else:
                    log.debug(f"Skipping inference save: model_id={model_id}, challenge_id={challenge_id}", indent=1)
        except Exception as e:
            log.warning(f"Failed to save inference record: {e}")
            import traceback
            traceback.print_exc()
    
    # Build response according to contract
    # Contract: results (required), optional: model, perform_time, challenge_id, message, error
    # Also include preprocess/postprocess metadata for UI display
    response = {
        'results': inference_results,  # Always a list
    }
    
    # Add optional fields only if they have values
    if model_doc:
        safe_model = dict(model_doc)
        # Make ObjectId JSON-safe
        if safe_model.get('_id'):
            safe_model['_id'] = str(safe_model['_id'])
        if safe_model.get('weights'):
            safe_model['weights'] = str(safe_model['weights'])
        response['model'] = safe_model
    
    if challenge_id:
        response['challenge_id'] = challenge_id
    
    if t1 - t0 > 0:
        response['perform_time'] = t1 - t0
    
    if message:
        response['message'] = message
    
    if error:
        response['error'] = error
    
    # Add preprocessing metadata for UI display
    if preprocess_meta:
        response['preprocess'] = preprocess_meta
    
    # Add postprocessing metadata for UI display
    if postprocess_meta:
        response['postprocess'] = postprocess_meta
    
    # Add preprocessed image for UI display (base64 encoded)
    # Only include if preprocessing was actually applied
    if preprocess_meta:
        try:
            processed_image_b64 = base64.b64encode(processed_img_bytes).decode('utf-8')
            response['processed_image'] = processed_image_b64
        except Exception as e:
            log.warning(f"Failed to encode processed image: {e}")
    
    return jsonify(response)

@app.route('/solve_hcaptcha_batch', methods=['POST'])
def solve_hcaptcha_batch():
    """
    Headless API endpoint for processing multiple images (tiles) in batch.
    
    Contract (per hcaptcha-rules.mdc):
    - Request: multiple image bytes (multipart/form-data 'images') + question string (form 'question')
    - Response: {
        'results': list of per-image entries, each with:
            - 'image_index': int (1-based)
            - 'results': list of detection dicts (each with 'bbox', 'class', 'confidence')
        'model': optional model info dict,
        'perform_time': optional float,
        'challenge_id': optional string,
        'message': optional string,
        'error': optional string
      }
    
    Detection dict format:
    - 'bbox': [x_min, y_min, x_max, y_max] in intrinsic image coordinates (model space)
    - 'class': predicted class label (str)
    - 'confidence': float between 0-1
    
    Intended usage:
    - Crawler (client/crawler.py): Sends multiple tile images from hCAPTCHA challenges
    """
    # Validate request
    images = request.files.getlist('images')
    if not images:
        return jsonify({
            'error': 'missing_images',
            'message': 'No images provided in request',
            'results': []
        }), 400
    
    question = request.form.get('question', '')
    
    # Identify config view (not used for solving now)
    config = get_model_config(question)
    import time
    t0 = time.time()
    
    # Process all images
    all_results = []
    image_names = []
    compressed_images = []
    
    # Find challenge_type and get model directly from it
    from .database import _find_challenge_type_for_question, get_model_by_id
    ct_doc = None
    model_doc = None
    challenge_type_matched = False
    single_challenge_id = None
    
    if question:
        ct_doc = _find_challenge_type_for_question(question)
        if ct_doc:
            challenge_type_matched = True
            challenge_type_id = ct_doc.get("challenge_type_id")
            log.info(f"Challenge type matched: {challenge_type_id}")
            
            # Get model directly from challenge_type document
            model_id = ct_doc.get("model_id")
            if model_id:
                model_doc = get_model_by_id(model_id)
                if model_doc:
                    log.info(f"Model found: {model_id}")
                else:
                    log.warning(f"challenge_type {challenge_type_id} has model_id {model_id}, but model not found in database")
            else:
                log.warning(f"challenge_type {challenge_type_id} has no model_id set")
            
            # Only save challenge if it matches a challenge_type
            single_challenge_id = f"ch-{int(t0)}"
            create_challenge(
                challenge_id=single_challenge_id,
                challenge_questions=question or "",
                challenge_img=[],  # Will be populated after preprocessing
                challenge_type_id=challenge_type_id,
            )
        else:
            # No matching challenge_type - don't save challenge
            log.info(f"No matching challenge_type for question: {question}")
            single_challenge_id = None
    else:
        # No question provided - don't save challenge
        log.info("No question provided - skipping challenge save")
    
    # Get preprocessing profile if model has preprocess_id
    preprocess_profile = None
    if model_doc:
        try:
            preprocess_profile = get_preprocess_for_model(model_doc)
        except Exception as e:
            log.warning(f"Preprocess profile retrieval error: {e}")
    
    # Get postprocess profile for inference (retrieve once, use for all images)
    postprocess_profile_retrieved = None
    if model_doc:
        try:
            postprocess_profile_retrieved = get_postprocess_for_model(model_doc)
        except Exception as e:
            log.warning(f"Postprocess profile retrieval error: {e}")
    
    # Process each image with preprocessing and inference
    all_results = []
    base_timestamp = int(time.time() * 1000)  # Base timestamp for all images in this batch
    message = None
    error = None
    
    for idx, img in enumerate(images, start=1):
        # Read raw bytes for compression and model use
        img_bytes = img.read()
        # Rewind the stream for any downstream consumer that might need it
        try:
            img.stream.seek(0)
        except Exception:
            pass
        
        image_names.append(img.filename)
        
        # Apply preprocessing if profile exists
        processed_img_bytes = img_bytes
        if preprocess_profile:
            try:
                processed_img_bytes, _, _ = apply_preprocess(img_bytes, preprocess_profile)
                log.debug(f"Preprocessed image {idx} successfully", indent=1)
            except Exception as e:
                log.warning(f"Preprocessing error for image {idx}: {e}")
        
        # Run inference on processed image with postprocess steps
        image_results = []
        inference_start_time = time.time()
        try:
            if model_doc:
                # Pass full postprocess profile for modular postprocessing
                postprocess_profile = None
                if postprocess_profile_retrieved:
                    postprocess_profile = {
                        'postprocess_id': postprocess_profile_retrieved.get('postprocess_id'),
                        'name': postprocess_profile_retrieved.get('name'),
                        'steps': postprocess_profile_retrieved.get('steps', [])
                    }
                
                image_results = solve_captcha(
                    processed_img_bytes,
                    question,
                    config,
                    postprocess_profile=postprocess_profile
                )
                # Normalize image_results to always be a list of detections
                # solve_captcha returns: list, dict with 'error', or dict with 'message'
                if isinstance(image_results, dict):
                    if 'error' in image_results:
                        log.warning(f"Inference error for image {idx}: {image_results['error']}")
                        image_results = []  # Return empty list on error
                        if error is None:
                            error = 'inference_error'
                            message = f"Inference failed for one or more images: {image_results.get('error', 'unknown error')}"
                    elif 'message' in image_results:
                        # No detections found - return empty list
                        log.debug(f"Inference message for image {idx}: {image_results.get('message')}", indent=1)
                        image_results = []
                    else:
                        # Unexpected dict format - treat as error
                        image_results = []
                        if error is None:
                            error = 'invalid_inference_result'
                            message = 'Inference returned unexpected format for one or more images'
                elif not isinstance(image_results, list):
                    # Invalid format - return empty list
                    image_results = []
                    if error is None:
                        error = 'invalid_inference_result'
                        message = 'Inference returned invalid format for one or more images'
            else:
                # No model document found
                image_results = []
                if error is None:
                    error = 'no model selected'
                    message = 'No model available for this challenge type'
            
            inference_end_time = time.time()
            inference_speed = inference_end_time - inference_start_time
            
            # Contract: results is list of entries with image_index (1-based) and results (list of detections)
            all_results.append({
                'image_index': idx,  # 1-based index per contract
                'results': image_results  # Always a list of detection dicts
            })
            
            # Save inference to inference collection only if no errors and has detections
            if image_results:
                try:
                    inference_id = f"inf-{base_timestamp}-{idx}"  # Use base timestamp and index for uniqueness
                    model_id = config.get('model_id') or (model_doc.get('model_id') if model_doc else None)
                    
                    # Ensure detection_results is in the correct format (list or dict)
                    detection_results = image_results
                    if not isinstance(detection_results, (list, dict)):
                        detection_results = []
                    
                    # Only save inference if challenge was saved (i.e., challenge_type matched)
                    if model_id and single_challenge_id:
                        created_inf = create_inference(
                            inference_id=inference_id,
                            inference_speed=inference_speed,
                            detection_results=detection_results,
                            model_id=model_id,
                            challenge_id=single_challenge_id
                        )
                        log.debug(f"Saved inference record for image {idx}: {inference_id}", indent=1)
                        # Save activity log entry for this image
                        try:
                            log_id = f"act-{base_timestamp}-{idx}"
                            create_activity_log(
                                log_id=log_id,
                                challenge_id=single_challenge_id,
                                model_id=model_id,
                                inference_id=inference_id
                            )
                            log.debug(f"Saved activity_log record for image {idx}: {log_id}", indent=1)
                        except Exception as e:
                            log.warning(f"Failed to save activity_log for image {idx}: {e}")
                    else:
                        if not single_challenge_id:
                            log.debug(f"Skipping inference save for image {idx}: challenge not saved (no matching challenge_type)", indent=1)
                        else:
                            log.debug(f"Skipping inference save for image {idx}: model_id={model_id}, challenge_id={single_challenge_id}", indent=1)
                except Exception as e:
                    log.warning(f"Failed to save inference record for image {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                
        except Exception as e:
            log.error(f"Inference error for image {idx}: {e}")
            # On error, add entry with empty results list
            all_results.append({
                'image_index': idx,
                'results': []  # Empty list on error
            })
            if error is None:
                error = 'inference_error'
                message = f"Inference failed for one or more images: {str(e)}"
            # Do not save inference record when there's an error
        
        # Compress each processed image and store in array
        try:
            compressed_data = zlib.compress(processed_img_bytes)
            compressed_images.append(Binary(compressed_data))
        except Exception as e:
            log.warning(f"Failed to compress image {img.filename}: {e}")
    
    # Update challenge with processed images
    if compressed_images:
        try:
            create_challenge(
                challenge_id=single_challenge_id,
                challenge_questions=question or "",
                challenge_img=compressed_images,
            )
        except Exception as e:
            log.warning(f"Failed to update challenge with images: {e}")
    
    t1 = time.time()
    
    # Determine message/error for challenge type mismatch
    if not challenge_type_matched:
        if message is None:
            message = "no match challenge type found"
        if error is None:
            error = "no match challenge type found"
        # If no match, return early without processing
        if not single_challenge_id:
            return jsonify({
                'error': error,
                'message': message,
                'results': [],
            })
    
    # Build response according to contract
    # Contract: results (required), optional: model, perform_time, challenge_id, message, error
    # Also include preprocess/postprocess metadata for UI display
    response = {
        'results': all_results,  # List of entries with image_index and results
    }
    
    # Add optional fields only if they have values
    if model_doc:
        safe_model = dict(model_doc)
        if safe_model.get('_id'):
            safe_model['_id'] = str(safe_model['_id'])
        if safe_model.get('weights'):
            safe_model['weights'] = str(safe_model['weights'])
        response['model'] = safe_model
    
    if single_challenge_id:
        response['challenge_id'] = single_challenge_id
    
    if t1 - t0 > 0:
        response['perform_time'] = t1 - t0
    
    if message:
        response['message'] = message
    
    if error:
        response['error'] = error
    
    # Add preprocessing metadata for UI display (from first image's preprocessing)
    if preprocess_profile:
        # Create preprocess metadata from profile
        preprocess_meta = {
            'preprocess_id': preprocess_profile.get('preprocess_id'),
            'applied_steps': [],  # Batch doesn't track individual steps per image
        }
        response['preprocess'] = preprocess_meta
    
    # Add postprocessing metadata for UI display
    if postprocess_profile_retrieved:
        postprocess_meta = {
            'postprocess_id': postprocess_profile_retrieved.get('postprocess_id'),
            'name': postprocess_profile_retrieved.get('name'),
            'steps': postprocess_profile_retrieved.get('steps', []),
        }
        response['postprocess'] = postprocess_meta
    
    return jsonify(response)

# =========================
# Model management endpoints
# =========================

@app.route('/models', methods=['POST'])
def create_or_update_model():
    """Create or update a model. Accepts multipart form-data with a file field `weights`."""
    model_id = request.form.get('model_id')
    model_name = request.form.get('model_name')
    is_active = request.form.get('is_active', 'false').lower() == 'true'
    # Metrics may come as strings; cast if present
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None
    results = {
        'precision': _to_float(request.form.get('precision')),
        'recall': _to_float(request.form.get('recall')),
        'f1_score': _to_float(request.form.get('f1_score')),
        'mAP50': _to_float(request.form.get('mAP50')),
        'AP5095': _to_float(request.form.get('AP5095')),
    }
    # Remove None values
    results = {k: v for k, v in results.items() if v is not None}

    if not model_id or not model_name:
        return jsonify({'error': 'model_id and model_name are required'}), 400

    weights_file = request.files.get('weights')
    
    if weights_file:
        # We MUST save to a temp file because the Flask request stream closes 
        # immediately after this function returns.
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"{model_id}_upload.pt")
        
        # Save the file stream to disk (fast)
        weights_file.save(temp_path)
        
        # 3. Define the background worker function
        def background_upload_task(f_path, m_id, m_name, res, active):
            log.info(f"[Background] Upload started for {m_id}")
            try:
                # Open the temp file and stream it to GridFS
                # This is the part that takes 10 minutes, but now it runs in the background
                with open(f_path, 'rb') as f_stream:
                    upsert_model(m_id, m_name, f_stream, res, is_active=active)
                log.success(f"[Background] Upload finished successfully for {m_id}")
            except Exception as e:
                log.error(f"[Background] Upload FAILED for {m_id}: {e}")
            finally:
                # Cleanup: Delete the temp file to free up space
                if os.path.exists(f_path):
                    os.remove(f_path)

        # 4. Start the background thread
        thread = threading.Thread(
            target=background_upload_task, 
            args=(temp_path, model_id, model_name, results, is_active)
        )
        thread.start()
        
        # 5. Return Success IMMEDIATELY
        return jsonify({
            'ok': True, 
            'model_id': model_id, 
            'status': 'uploading_in_background',
            'message': 'File accepted. Uploading in background. Please wait for it to appear in the model list.'
        })
    
    else:
        # Metadata-only update (fast, no need for background)
        upsert_model(model_id, model_name, None, results, is_active=is_active)
    return jsonify({'ok': True, 'model_id': model_id})

@app.route('/models', methods=['GET'])
def get_models():
    items = list_models(limit=int(request.args.get('limit', 50)))
    # Convert ObjectId to str for JSON
    for it in items:
        if it.get('_id'):
            it['_id'] = str(it['_id'])
        if it.get('weights'):
            it['weights'] = str(it['weights'])
    return jsonify({'items': items})

if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT)