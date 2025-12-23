# hCAPTCHA Solver

An automated hCAPTCHA solving system powered by YOLO object detection models. The system consists of three main components: a headless Flask API service for model inference, a Selenium-based browser automation client for crawling and solving challenges, and a Streamlit demo UI for visualizstion and testing.

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

### 1. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies

```bash (Recommended)
pip install Flask pymongo gridfs ultralytics selenium webdriver-manager streamlit pillow numpy opencv-python pandas pyyaml plotly python-dotenv requests
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

### 3. Configure Environment Variables (if download from github)

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

```bash
# From project root
streamlit run streamlit_demo/main.py
```

The UI will open in your browser at `http://localhost:8501`.

