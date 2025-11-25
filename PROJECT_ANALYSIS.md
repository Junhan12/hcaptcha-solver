# hCaptcha Solver - Project Architecture Analysis

## Overview
This is an automated hCaptcha solving system that uses YOLO object detection models to identify and click on challenge elements. The system consists of three main components: a Flask API backend, Selenium-based web automation clients, and a Streamlit demo interface.

---

## Project Structure

### 1. **app/** - Core Application Backend

#### `app/__init__.py`
- **Purpose**: Package initialization and public API exports
- **Exports**:
  - `flask_app`: Flask application instance
  - `create_app()`: Factory function for WSGI servers
  - `solve_captcha`: Core solver function
  - Database helpers: `get_model_config`, `validate_question_and_get_model`, `upsert_model`, etc.
- **Dependencies**: Imports from `solver`, `database`, `helper`, `api_gateway`

#### `app/config.py`
- **Purpose**: Centralized configuration management
- **Key Configurations**:
  - MongoDB connection string (`MONGO_URI`)
  - Flask settings (debug, port, host)
  - API timeouts and retries
  - Model configuration defaults
  - Preprocessing/postprocessing timeouts
  - Database connection pooling
  - Logging configuration
- **Usage**: Loaded by `database.py` and other modules via environment variables

#### `app/database.py`
- **Purpose**: MongoDB database operations and data layer
- **Key Collections**:
  1. **`model`**: Stores YOLO model metadata and weights (GridFS)
  2. **`challenge_type`**: Maps questions to models via keywords
  3. **`challenge`**: Stores challenge images and questions
  4. **`inference`**: Stores inference results and performance metrics
  5. **`activity_log`**: Tracks challenge → model → inference relationships
  6. **`preprocess_profile`**: Image preprocessing configurations
  7. **`postprocess_profile`**: Post-processing configurations (confidence/IoU thresholds)

- **Core Functions**:
  - `_find_challenge_type_for_question()`: Matches questions to challenge types (requires ≥2 keyword matches)
  - `validate_question_and_get_model()`: Gets model based on challenge type matching
  - `get_model_for_challenge()`: Retrieves model linked via challenge → challenge_type → model
  - `upsert_model()`: Stores model weights in GridFS, metadata in `model` collection
  - `create_challenge()`: Saves challenge with compressed images
  - `create_inference()`: Saves inference results with timestamps
  - `get_preprocess_for_model()` / `get_postprocess_for_model()`: Retrieves processing profiles

- **Dependencies**: `config.py` for MongoDB URI

#### `app/preprocess.py`
- **Purpose**: Image preprocessing pipeline with modular operations
- **Operations Supported**:
  - `bilateral`: Edge-preserving smoothing
  - `median`: Noise reduction
  - `gaussian`: Gaussian blur
  - `nlmeans`: Non-local means denoising
  - `laplacian`: Edge detection
  - `addweighted`: Image blending
  - `sharpen`: Sharpening filter
  - `grayscale`: Color to grayscale conversion

- **Key Function**: `apply_preprocess(img_bytes, preprocess_profile)`
  - Takes raw image bytes and a profile with steps
  - Applies operations sequentially
  - Returns processed bytes and applied steps metadata

- **Dependencies**: OpenCV (`cv2`), PIL, numpy

#### `app/solver.py`
- **Purpose**: Core YOLO model inference engine
- **Key Function**: `solve_captcha(image, question, config, postprocess_steps=None)`
  - Loads model weights from database (GridFS) or by model_id
  - Converts image bytes to PIL/numpy format
  - Handles grayscale/RGBA → RGB conversion
  - Runs YOLO inference with confidence/IoU thresholds
  - Returns list of detections: `[{'class': str, 'confidence': float, 'bbox': [x1, y1, x2, y2]}, ...]`
  - Returns `{'error': str}` or `{'message': str, 'detections': []}` on failure/no detections

- **Dependencies**: 
  - `ultralytics` (YOLO)
  - `app.database.download_weights_bytes` for model loading

#### `app/api_gateway.py`
- **Purpose**: Flask REST API endpoints
- **Endpoints**:

  1. **`POST /solve_hcaptcha`** (Single image)
     - Receives: image file + question
     - Flow:
       1. Checks if question matches a `challenge_type` (≥2 keywords)
       2. If matched: creates `challenge` record, retrieves model
       3. Applies preprocessing if model has `preprocess_id`
       4. Retrieves postprocess profile (confidence/IoU thresholds)
       5. Calls `solve_captcha()` with processed image
       6. Saves inference record if successful
       7. Creates activity log entry
     - Returns: detections, processed image (base64), metadata

  2. **`POST /solve_hcaptcha_batch`** (Multiple images/tiles)
     - Receives: multiple image files + question
     - Flow: Similar to single, but processes each image separately
     - Stores all images as compressed array in challenge
     - Returns: array of results per image

  3. **`POST /models`** (Model management)
     - Creates/updates model with weights file
     - Stores weights in GridFS

  4. **`GET /models`** (List models)
     - Returns all models with metadata

- **Dependencies**: 
  - `solver.solve_captcha`
  - `database.*` functions
  - `preprocess.apply_preprocess`

#### `app/helper.py`
- **Purpose**: Utility functions
- **Key Function**: `decompress_image_to_base64(compressed_binary)`
  - Decompresses zlib-compressed Binary data from MongoDB
  - Converts to base64 string for display
  - Handles both single Binary and arrays

---

### 2. **client/** - Web Automation Clients

#### `client/crawler.py`
- **Purpose**: Selenium-based web crawler that extracts hCaptcha challenges and sends them to API
- **Key Functions**:

  - `run_crawl_once()`: Main orchestration function
    1. Launches Chrome browser
    2. Navigates to hCaptcha demo site
    3. Clicks checkbox to trigger challenge
    4. **Refresh loop**: Checks if question matches `challenge_type` (via `check_question_matches_challenge_type()`)
    5. If no match: clicks refresh button (up to 20 attempts)
    6. Extracts challenge images (canvas or div tiles)
    7. Sends to API endpoints
    8. Handles multi-crumb challenges (multiple sequential challenges)

  - `send_canvas_images()`: 
    - Extracts canvas elements via JavaScript (`toDataURL`)
    - Sends each canvas to `/solve_hcaptcha`
    - Calls `perform_clicks()` from `clicker.py` to automate clicking

  - `send_nested_div_images()`:
    - Finds div elements with background images
    - Downloads images from URLs
    - Handles 9-tile vs 10-tile layouts (10th is sample image)
    - Sends batch to `/solve_hcaptcha_batch`
    - Calls `perform_clicks()` for tile clicking

  - `check_question_matches_challenge_type()`:
    - Uses `_find_challenge_type_for_question()` from database
    - Fallback: sends test API request if database unavailable

- **Dependencies**: 
  - `client.clicker.perform_clicks`
  - `app.database._find_challenge_type_for_question` (optional)
  - Selenium, requests

#### `client/clicker.py`
- **Purpose**: Translates inference results into Selenium clicks
- **Key Functions**:

  - `perform_clicks()`: Unified entry point
    - Mode: `"canvas"` or `"tiles"`
    - Routes to appropriate clicking function

  - `click_canvas_from_response()`:
    - Extracts detections from API response
    - Calculates click coordinates from bounding boxes
    - Handles canvas coordinate scaling (intrinsic vs displayed size)
    - Uses ActionChains or JavaScript events for clicking
    - Handles stale element references (canvas re-renders after clicks)

  - `click_tiles_from_batch_response()`:
    - Processes batch API results
    - Determines which tiles have positive detections
    - Clicks tile div elements directly (not bounding boxes)

- **Dependencies**: Selenium WebDriver

---

### 3. **streamlit_demo/** - Demo Interface

#### `streamlit_demo/main.py`
- **Purpose**: Streamlit web UI for testing and demonstration
- **Pages**:

  1. **"Upload -> API"**:
     - Manual image upload
     - Question input
     - Displays processed image with bounding boxes
     - Shows detection results table
     - Displays preprocessing/postprocessing metadata

  2. **"Crawl -> API"**:
     - Triggers `run_crawl_once()` from crawler
     - Displays results with images and detections
     - Handles multi-crumb and batch result visualization

  3. **"Create/Update Model"**:
     - Form to upload model weights
     - Set model metrics (precision, recall, F1, mAP50, AP5095)
     - List existing models

- **Dependencies**: 
  - `client.crawler.run_crawl_once`
  - `app.helper.decompress_image_to_base64`
  - Streamlit, requests, PIL

---

## Data Flow

### Challenge Processing Flow

```
1. Crawler extracts question from hCaptcha
   ↓
2. Check if question matches challenge_type (≥2 keywords)
   ↓
3. If match: Extract images (canvas/divs)
   ↓
4. Send to API: /solve_hcaptcha or /solve_hcaptcha_batch
   ↓
5. API: Find challenge_type → Get model → Get preprocess/postprocess profiles
   ↓
6. Apply preprocessing (if configured)
   ↓
7. Run YOLO inference with postprocess thresholds
   ↓
8. Save challenge, inference, and activity_log records
   ↓
9. Return detections to crawler
   ↓
10. Clicker translates detections to clicks
    ↓
11. Selenium performs clicks on canvas/tiles
```

### Database Relationships

```
challenge_type
  ├─ keywords: ["keyword1", "keyword2", ...]  (≥2 must match question)
  ├─ method: "drag" | "click"
  └─ model_id → model.model_id

model
  ├─ model_id (unique)
  ├─ model_name
  ├─ weights (GridFS ObjectId)
  ├─ preprocess_id → preprocess_profile.preprocess_id
  ├─ postprocess_id → postprocess_profile.postprocess_id
  └─ is_active (boolean)

challenge
  ├─ challenge_id (unique)
  ├─ challenge_questions (text)
  ├─ challenge_img (compressed Binary or array)
  └─ challenge_type_id → challenge_type.challenge_type_id

inference
  ├─ inference_id (unique)
  ├─ detection_results (list/dict)
  ├─ inference_speed (seconds)
  ├─ model_id → model.model_id
  └─ challenge_id → challenge.challenge_id

activity_log
  ├─ log_id (unique)
  ├─ challenge_id → challenge.challenge_id
  ├─ model_id → model.model_id
  └─ inference_id → inference.inference_id
```

---

## Key Design Patterns

1. **Keyword-Based Challenge Type Matching**: Questions are matched to challenge types by requiring ≥2 keywords to appear in the question text (case-insensitive substring matching).

2. **Modular Preprocessing**: Preprocessing operations are registered in a dictionary and applied sequentially based on profile configuration.

3. **GridFS for Model Weights**: Large model weight files are stored in MongoDB GridFS, referenced by ObjectId in the model document.

4. **Compressed Image Storage**: Challenge images are compressed with zlib before storage to save space.

5. **Error Handling**: Inference errors are not saved to the database; only successful inferences are recorded.

6. **Multi-Crumb Support**: The crawler handles sequential challenges (crumbs) by processing each separately and clicking submit buttons between them.

7. **Refresh Loop**: The crawler automatically refreshes challenges until it finds one that matches a configured challenge_type.

---

## Dependencies Summary

- **Backend**: Flask, pymongo, gridfs, ultralytics (YOLO), opencv-python, PIL, numpy
- **Client**: Selenium, webdriver-manager, requests
- **UI**: Streamlit
- **Database**: MongoDB (with GridFS)

---

## Configuration Requirements

1. **MongoDB**: Connection string in `.env` as `MONGO_URI`
2. **Challenge Types**: Must be pre-configured in `challenge_type` collection with keywords and model_id
3. **Models**: Must be uploaded via API or Streamlit interface before use
4. **Preprocess/Postprocess Profiles**: Must be configured in respective collections if models reference them

---

## Current Limitations / TODO Items

Based on `TO-DO list.md`:
1. Database URL configuration (partially done via config.py)
2. MongoDB table creation (indexes handled by `ensure_indexes()`)
3. Logical algorithm for question validation (keyword matching implemented)
4. Model configuration (implemented via API)
5. Logging (basic print statements, not structured logging)
6. **Validation logic for challenge_type ct-002**: Not yet implemented (see TO-DO list for requirements)

