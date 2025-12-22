# hCAPTCHA Solver

An automated hCAPTCHA solving system powered by YOLO object detection models. The system consists of three main components: a headless Flask API service for model inference, a Selenium-based browser automation client for crawling and solving challenges, and a Streamlit demo UI for visualization and testing.

## Project Structure

```
solver/
├── app/                    # API service layer
│   ├── api_gateway.py     # Flask HTTP endpoints
│   ├── solver.py          # Model inference orchestration
│   ├── database.py        # MongoDB operations
│   ├── preprocess.py      # Image preprocessing pipeline
│   ├── postprocess.py     # Detection postprocessing
│   ├── evaluator.py       # Model evaluation metrics
│   └── config.py          # Configuration management
├── client/                 # Browser automation
│   ├── crawler.py         # hCAPTCHA page navigation & image capture
│   └── clicker.py         # Detection-to-click coordinate translation
└── streamlit_demo/        # Web UI for debugging & testing
    ├── main.py            # Main Streamlit app entry point
    └── page_modules/       # Individual page modules
```

## Prerequisites

- **Python 3.10+**
- **MongoDB** (local or remote instance)
- **Chrome/Chromium** browser (for Selenium automation)
- **ChromeDriver** (automatically managed by webdriver-manager)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd solver
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install flask pymongo gridfs ultralytics selenium webdriver-manager streamlit pillow numpy opencv-python pandas pyyaml plotly python-dotenv requests
```

Or install from a requirements file (if available):

```bash
pip install -r requirements.txt
```

### Core Dependencies

- **flask** - Web framework for API service
- **pymongo** - MongoDB driver
- **gridfs** - MongoDB GridFS for storing model weights
- **ultralytics** - YOLO model inference
- **selenium** - Browser automation
- **webdriver-manager** - Automatic ChromeDriver management
- **streamlit** - Web UI framework
- **pillow** (PIL) - Image processing
- **numpy** - Numerical operations
- **opencv-python** (cv2) - Advanced image preprocessing
- **pandas** - Data manipulation
- **pyyaml** - YAML file parsing
- **plotly** - Interactive visualizations (optional)
- **python-dotenv** - Environment variable management
- **requests** - HTTP client

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017/
# Or for MongoDB Atlas:
# MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/

# Flask API Configuration
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=False

# API Timeout (seconds)
API_TIMEOUT=300

# Database Configuration
DB_CONNECTION_TIMEOUT=10000
DB_MAX_POOL_SIZE=50
DB_MIN_POOL_SIZE=10

# Logging
LOG_LEVEL=INFO
LOG_FILE=app.log
```

## Running the Project

### 1. Start MongoDB

Ensure MongoDB is running locally or configure a remote connection in `.env`:

```bash
# Local MongoDB (if installed)
mongod
```

### 2. Start the Flask API Service

The API service handles model inference, preprocessing, postprocessing, and database operations.

```bash
# From project root
python -m app.api_gateway
```

Or:

```bash
cd app
python api_gateway.py
```

The API will start on `http://0.0.0.0:5000` (or the port specified in `.env`).

**API Endpoints:**
- `POST /solve_hcaptcha` - Process single image challenge
- `POST /solve_hcaptcha_batch` - Process multiple tile images
- `POST /models` - Create or update model
- `GET /models` - List all models

### 3. Run the Streamlit Demo UI

The Streamlit app provides a web interface for:
- Auto-crawling hCAPTCHA datasets
- Data preprocessing and augmentation
- Model training workflows
- Model upload and management
- Model evaluation with metrics
- Interactive inference testing

```bash
# From project root
streamlit run streamlit_demo/main.py
```

The UI will open in your browser at `http://localhost:8501`.

### 4. Run the Browser Automation Client

The crawler automates hCAPTCHA challenges by:
1. Navigating to the challenge page
2. Clicking the checkbox
3. Extracting challenge images (canvas or tiles)
4. Sending images to the API for inference
5. Performing automatic clicks based on detections

```bash
# From project root
python -m client.crawler
```

Or use the Streamlit UI's "Auto Crawler, Solver, and Clicker" section for interactive automation.

## Usage Examples

### Using the API Directly

**Single Image Inference:**

```python
import requests

url = "http://localhost:5000/solve_hcaptcha"
files = {'image': open('challenge.png', 'rb')}
data = {'question': 'Select all images with traffic lights'}

response = requests.post(url, files=files, data=data)
result = response.json()

# Access detections
detections = result['results']
for det in detections:
    bbox = det['bbox']  # [x_min, y_min, x_max, y_max]
    class_name = det['class']
    confidence = det['confidence']
```

**Batch Inference (Multiple Tiles):**

```python
import requests

url = "http://localhost:5000/solve_hcaptcha_batch"
files = [('images', open(f'tile_{i}.png', 'rb')) for i in range(9)]
data = {'question': 'Select all images with buses'}

response = requests.post(url, files=files, data=data)
result = response.json()

# Access per-image results
for entry in result['results']:
    image_index = entry['image_index']  # 1-based
    detections = entry['results']
```

### Using the Crawler Programmatically

```python
from client.crawler import run_crawl_once

# Run a single crawl session
result = run_crawl_once(
    headless=False,  # Set to True for headless mode
    max_refreshes=3
)
```

## Database Schema

The system uses MongoDB with the following collections:

- **model** - YOLO model metadata and weights (stored in GridFS)
- **challenge_type** - Challenge question patterns and model mappings
- **challenge** - Captured challenge images and metadata
- **inference** - Inference results and performance metrics
- **activity_log** - Activity tracking
- **preprocess_profile** - Preprocessing pipeline configurations
- **postprocess_profile** - Postprocessing pipeline configurations

## Model Management

### Upload a Model via API

```python
import requests

url = "http://localhost:5000/models"
files = {'weights': open('model.pt', 'rb')}
data = {
    'model_id': 'model-001',
    'model_name': 'YOLOv8 Traffic Lights',
    'is_active': 'true'
}

response = requests.post(url, files=files, data=data)
```

### Create Challenge Type Mapping

Use the Streamlit UI or MongoDB directly to create challenge types that map questions to models:

```python
from app.database import _db

_db.challenge_type.insert_one({
    'challenge_type_id': 'ct-001',
    'keywords': ['traffic light', 'traffic lights'],
    'model_id': 'model-001'
})
```

## Configuration

### API Configuration

Edit `app/config.py` or set environment variables:

- `FLASK_HOST` - API host (default: `0.0.0.0`)
- `FLASK_PORT` - API port (default: `5000`)
- `API_TIMEOUT` - Request timeout in seconds (default: `300`)

### Browser Automation Configuration

The crawler uses Chrome/Chromium with Selenium. ChromeDriver is automatically managed by `webdriver-manager`.

## Troubleshooting

### MongoDB Connection Issues

- Verify MongoDB is running: `mongosh` or check service status
- Check `MONGO_URI` in `.env` file
- Ensure network access if using remote MongoDB

### ChromeDriver Issues

- `webdriver-manager` should automatically download the correct ChromeDriver version
- If issues persist, manually install ChromeDriver and add to PATH

### Model Loading Errors

- Ensure model weights are properly uploaded to MongoDB GridFS
- Check model file format (should be `.pt` YOLO weights)
- Verify `model_id` exists in database

### Import Errors

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Activate virtual environment if using one
- Check Python version (3.10+ required)

## Development

### Project Architecture

The system follows a clean separation of concerns:

- **app/** - API, inference, and database logic (no Selenium/UI dependencies)
- **client/** - Browser automation (communicates with app via HTTP)
- **streamlit_demo/** - UI layer (communicates with app via HTTP)

### Code Style

- Follow PEP 8 conventions
- Keep line length ≤ 80 characters where practical
- Use descriptive variable names
- Add comments for complex logic

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable]
