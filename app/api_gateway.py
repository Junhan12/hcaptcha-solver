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

app = Flask(__name__)

# =========================
# Headless API endpoints (for crawler, clicker, and Streamlit upload only)
# =========================

@app.route('/solve_hcaptcha', methods=['POST'])
def solve_hcaptcha():
    """
    Headless API endpoint for processing single image challenges.
    
    Intended usage:
    - Crawler (client/crawler.py): Sends canvas images from hCAPTCHA challenges
    - Clicker (client/clicker.py): Sends images for processing before clicking
    - Streamlit upload page: Manual image upload for testing
    
    This endpoint is NOT intended for general API usage.
    """
    img = request.files['image']
    question = request.form.get('question')
    
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
    
    if question:
        ct_doc = _find_challenge_type_for_question(question)
        if ct_doc:
            challenge_type_matched = True
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
            )
        else:
            # No matching challenge_type - don't save challenge
            print(f"No matching challenge_type for question: {question}")
            challenge_id = None
    else:
        # No question provided - don't save challenge
        print("No question provided - skipping challenge save")
    
    # Retrieve model based on challenge_type.model_id (only if challenge was created)
    model_doc = None
    if challenge_id:
        model_doc = get_model_for_challenge(challenge_id)
    
    # Apply preprocessing if model has preprocess_id
    preprocess_meta = None
    processed_img_bytes = img_bytes
    if model_doc:
        try:
            profile = get_preprocess_for_model(model_doc)
            if profile:
                processed_img_bytes, applied_steps = apply_preprocess(img_bytes, profile)
                preprocess_meta = {
                    'preprocess_id': profile.get('preprocess_id'),
                    'applied_steps': applied_steps,
                }
                # Use processed bytes for storage
                img_bytes = processed_img_bytes
        except Exception as e:
            print(f"Preprocessing error: {e}")
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
            print(f"Postprocess retrieval error: {e}")
            postprocess_meta = None
    
    # Determine message when no matching challenge_type
    message = None
    if not challenge_type_matched:
        message = "no match challenge type found"
        # If no match, return early without processing
        if not challenge_id:
            return jsonify({
                'error': 'no match challenge type found',
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
            # Handle new return format: could be list, dict with 'error', or dict with 'message'
            if isinstance(inference_results, dict):
                if 'error' in inference_results:
                    print(f"Inference error: {inference_results['error']}")
                    inference_results = {'error': inference_results['error']}
                elif 'message' in inference_results:
                    print(f"Inference message: {inference_results['message']}")
                    inference_results = inference_results.get('detections', [])
        else:
            # No model document found
            inference_results = {'error': 'no model selected'}
    except Exception as e:
        print(f"Inference error: {e}")
        inference_results = {'error': str(e)}
    
    inference_end_time = time.time()
    inference_speed = inference_end_time - inference_start_time
    t1 = time.time()
    
    # Save inference to inference collection only if no errors
    has_error = False
    if isinstance(inference_results, dict) and 'error' in inference_results:
        has_error = True
        print(f"Skipping inference save due to error: {inference_results['error']}")
    
    if not has_error:
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
                print(f"Saved inference record: {inference_id}")
                # Save activity log entry
                try:
                    log_id = f"act-{int(time.time() * 1000)}"
                    create_activity_log(
                        log_id=log_id,
                        challenge_id=challenge_id,
                        model_id=model_id,
                        inference_id=inference_id
                    )
                    print(f"Saved activity_log record: {log_id}")
                except Exception as e:
                    print(f"Failed to save activity_log: {e}")
            else:
                if not challenge_id:
                    print(f"Skipping inference save: challenge not saved (no matching challenge_type)")
                else:
                    print(f"Skipping inference save: model_id={model_id}, challenge_id={challenge_id}")
        except Exception as e:
            print(f"Failed to save inference record: {e}")
            import traceback
            traceback.print_exc()
    # Convert processed image to base64 for display
    processed_image_base64 = None
    try:
        processed_image_base64 = base64.b64encode(processed_img_bytes).decode('utf-8')
    except Exception as e:
        print(f"Failed to encode processed image: {e}")
    
    # Respond
    safe_model = None
    if model_doc:
        safe_model = dict(model_doc)
        # Make ObjectId JSON-safe
        if safe_model.get('_id'):
            safe_model['_id'] = str(safe_model['_id'])
        if safe_model.get('weights'):
            safe_model['weights'] = str(safe_model['weights'])
    return jsonify({
        'results': inference_results,
        'processed_image': processed_image_base64,
        'perform_time': t1-t0,
        'challenge_id': challenge_id,
        'model': safe_model,
        'preprocess': preprocess_meta,
        'postprocess': postprocess_meta,
        'message': message,
    })

@app.route('/solve_hcaptcha_batch', methods=['POST'])
def solve_hcaptcha_batch():
    """
    Headless API endpoint for processing multiple images (tiles) in batch.
    
    Intended usage:
    - Crawler (client/crawler.py): Sends multiple tile images from hCAPTCHA challenges
    - Clicker (client/clicker.py): Sends batch images for processing before clicking
    
    This endpoint is NOT intended for general API usage.
    Stores images as an array of compressed Binary data.
    """
    images = request.files.getlist('images')
    question = request.form.get('question')
    
    if not images:
        return jsonify({'error': 'No images provided'}), 400
    
    # Identify config view (not used for solving now)
    config = get_model_config(question)
    import time
    t0 = time.time()
    
    # Process all images
    all_results = []
    image_names = []
    compressed_images = []
    
    # Check if question matches any challenge_type before saving
    from .database import _find_challenge_type_for_question
    challenge_type_matched = False
    single_challenge_id = None
    
    if question:
        ct_doc = _find_challenge_type_for_question(question)
        if ct_doc:
            challenge_type_matched = True
            # Only save challenge if it matches a challenge_type
            single_challenge_id = f"ch-{int(t0)}"
            create_challenge(
                challenge_id=single_challenge_id,
                challenge_questions=question or "",
                challenge_img=[],  # Will be populated after preprocessing
            )
        else:
            # No matching challenge_type - don't save challenge
            print(f"No matching challenge_type for question: {question}")
            single_challenge_id = None
    else:
        # No question provided - don't save challenge
        print("No question provided - skipping challenge save")
    
    # Retrieve model once based on derived challenge_type (only if challenge was created)
    model_doc = None
    if single_challenge_id:
        model_doc = get_model_for_challenge(single_challenge_id)
    
    # Get preprocessing profile if model has preprocess_id
    preprocess_profile = None
    if model_doc:
        try:
            preprocess_profile = get_preprocess_for_model(model_doc)
        except Exception as e:
            print(f"Preprocess profile retrieval error: {e}")
    
    # Get postprocess profile for inference (retrieve once, use for all images)
    postprocess_profile_retrieved = None
    if model_doc:
        try:
            postprocess_profile_retrieved = get_postprocess_for_model(model_doc)
        except Exception as e:
            print(f"Postprocess profile retrieval error: {e}")
    
    # Process each image with preprocessing and inference
    processed_images_base64 = []
    all_results = []
    base_timestamp = int(time.time() * 1000)  # Base timestamp for all images in this batch
    
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
        print("start preprocessing image")
        if preprocess_profile:
            print("start applying preprocessing")
            try:
                processed_img_bytes, _ = apply_preprocess(img_bytes, preprocess_profile)
                print(f"Preprocessed image {idx} successfully")
            except Exception as e:
                print(f"Preprocessing error for image {idx}: {e}")
        
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
                # Handle new return format: could be list, dict with 'error', or dict with 'message'
                if isinstance(image_results, dict):
                    if 'error' in image_results:
                        print(f"Inference error for image {idx}: {image_results['error']}")
                        image_results = {'error': image_results['error']}
                    elif 'message' in image_results:
                        print(f"Inference message for image {idx}: {image_results['message']}")
                        image_results = image_results.get('detections', [])
            else:
                # No model document found
                image_results = {'error': 'no model selected'}
            
            inference_end_time = time.time()
            inference_speed = inference_end_time - inference_start_time
            
            all_results.append({
                'image_index': idx,
                'image_name': img.filename,
                'results': image_results
            })
            
            # Save inference to inference collection only if no errors
            has_error = False
            if isinstance(image_results, dict) and 'error' in image_results:
                has_error = True
                print(f"Skipping inference save for image {idx} due to error: {image_results['error']}")
            
            if not has_error:
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
                        print(f"Saved inference record for image {idx}: {inference_id}")
                        # Save activity log entry for this image
                        try:
                            log_id = f"act-{base_timestamp}-{idx}"
                            create_activity_log(
                                log_id=log_id,
                                challenge_id=single_challenge_id,
                                model_id=model_id,
                                inference_id=inference_id
                            )
                            print(f"Saved activity_log record for image {idx}: {log_id}")
                        except Exception as e:
                            print(f"Failed to save activity_log for image {idx}: {e}")
                    else:
                        if not single_challenge_id:
                            print(f"Skipping inference save for image {idx}: challenge not saved (no matching challenge_type)")
                        else:
                            print(f"Skipping inference save for image {idx}: model_id={model_id}, challenge_id={single_challenge_id}")
                except Exception as e:
                    print(f"Failed to save inference record for image {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                
        except Exception as e:
            print(f"Inference error for image {idx}: {e}")
            all_results.append({
                'image_index': idx,
                'image_name': img.filename,
                'results': {'error': str(e)}
            })
            # Do not save inference record when there's an error
        
        # Convert processed image to base64 for display
        try:
            processed_img_b64 = base64.b64encode(processed_img_bytes).decode('utf-8')
            processed_images_base64.append(processed_img_b64)
        except Exception as e:
            print(f"Failed to encode image {idx}: {e}")
            processed_images_base64.append(None)
        
        # Compress each processed image and store in array
        try:
            compressed_data = zlib.compress(processed_img_bytes)
            compressed_images.append(Binary(compressed_data))
        except Exception as e:
            print(f"Failed to compress image {img.filename}: {e}")
    
    # Update challenge with processed images
    if compressed_images:
        try:
            create_challenge(
                challenge_id=single_challenge_id,
                challenge_questions=question or "",
                challenge_img=compressed_images,
            )
        except Exception as e:
            print(f"Failed to update challenge with images: {e}")
    
    t1 = time.time()
    
    # Prepare response data
    safe_model = None
    preprocess_meta = None
    postprocess_meta = None
    
    if model_doc:
        safe_model = dict(model_doc)
        if safe_model.get('_id'):
            safe_model['_id'] = str(safe_model['_id'])
        if safe_model.get('weights'):
            safe_model['weights'] = str(safe_model['weights'])
        
        # Get preprocess metadata
        if preprocess_profile:
            preprocess_meta = {
                'preprocess_id': preprocess_profile.get('preprocess_id'),
                'name': preprocess_profile.get('name'),
            }
        
        # Get postprocess metadata (use already retrieved postprocess_profile_retrieved)
        if postprocess_profile_retrieved:
            postprocess_meta = {
                'postprocess_id': postprocess_profile_retrieved.get('postprocess_id'),
                'name': postprocess_profile_retrieved.get('name'),
                'steps': postprocess_profile_retrieved.get('steps', []),  # List of operations
            }

    message = None
    if not challenge_type_matched:
        message = "no match challenge type found"
        # If no match, return early without processing
        if not single_challenge_id:
            return jsonify({
                'error': 'no match challenge type found',
                'message': message,
                'results': [],
            })

    return jsonify({
        'results': all_results,
        'processed_images': processed_images_base64,
        'perform_time': t1-t0,
        'images_processed': len(images),
        'challenge_id': single_challenge_id,
        'model': safe_model,
        'preprocess': preprocess_meta,
        'postprocess': postprocess_meta,
        'message': message,
    })

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
            print(f"--- [Background] Upload started for {m_id} ---")
            try:
                # Open the temp file and stream it to GridFS
                # This is the part that takes 10 minutes, but now it runs in the background
                with open(f_path, 'rb') as f_stream:
                    upsert_model(m_id, m_name, f_stream, res, is_active=active)
                print(f"--- [Background] Upload finished successfully for {m_id} ---")
            except Exception as e:
                print(f"--- [Background] Upload FAILED for {m_id}: {e} ---")
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