"""
Shared utilities and imports for Streamlit demo.
"""
import os
import sys
import base64
import time
import io
import json
import importlib
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is on sys.path to import client and app modules
_this_dir = os.path.dirname(__file__)
_project_root = os.path.abspath(os.path.join(_this_dir, '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import app modules with error handling for Streamlit caching issues
from client.crawler import run_crawl_once
from app import decompress_image_to_base64
from app.config import API_TIMEOUT
from app.database import (
    list_models, 
    get_model_by_id, 
    get_preprocess_for_model, 
    get_postprocess_for_model, 
    upsert_model,
    get_preprocess_profile,
    get_postprocess_profile,
)

# Import delete_model with error handling
try:
    from app.database import delete_model
except ImportError:
    import app.database
    importlib.reload(app.database)
    from app.database import delete_model

from app.solver import solve_captcha

# Import cache functions with error handling
try:
    from app.solver import clear_model_cache, get_cache_info
except ImportError:
    import app.solver
    importlib.reload(app.solver)
    from app.solver import clear_model_cache, get_cache_info

from app.preprocess import apply_preprocess

# Import evaluator functions with error handling
try:
    from app.evaluator import evaluate_model, load_validation_dataset, parse_data_yaml
except ImportError:
    import app.evaluator
    importlib.reload(app.evaluator)
    from app.evaluator import evaluate_model, load_validation_dataset, parse_data_yaml

import pandas as pd
import yaml

# Try to import plotly for visualization
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# Helper functions to list preprocessing and postprocessing profiles
def list_preprocess_profiles(limit=100):
    """List all preprocessing profiles from MongoDB."""
    try:
        from app.database import _db, _db_available
        if not _db_available():
            return []
        cursor = _db.preprocess_profile.find({}).sort("preprocess_id", 1).limit(int(limit))
        return list(cursor)
    except Exception as e:
        print(f"Error listing preprocess profiles: {e}")
        return []


def list_postprocess_profiles(limit=100):
    """List all postprocessing profiles from MongoDB."""
    try:
        from app.database import _db, _db_available
        if not _db_available():
            return []
        cursor = _db.postprocess_profile.find({}).sort("postprocess_id", 1).limit(int(limit))
        return list(cursor)
    except Exception as e:
        print(f"Error listing postprocess profiles: {e}")
        return []

