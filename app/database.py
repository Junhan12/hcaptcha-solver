from pymongo import errors
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from .config import MONGO_URI, DEFAULT_MODEL_CONFIG
from bson import ObjectId
from datetime import datetime
try:
    import gridfs
except Exception:
    gridfs = None  # optional; only needed for model weights

_db = None
_uri = MONGO_URI or os.getenv("MONGO_URI")
if _uri:
    try:
        # Use stable Server API for Atlas; works locally as well
        client = MongoClient(
            _uri,
            server_api=ServerApi('1'),
            serverSelectionTimeoutMS=2000,
        )
        # Trigger server selection immediately
        client.admin.command("ping")

        # Create the model bucket if it doesn't exist
        _db = client["hcaptcha"]
        # Use GridFS with collection/bucket name "model" -> model.files/model.chunks
        _fs = gridfs.GridFS(_db, collection="model") if gridfs is not None else None
    except Exception:
        _db = None
        _fs = None
else:
    _db = None
    _fs = None


def _db_available():
    return _db is not None


def _find_challenge_type_for_question(question):
    """Return the first matching challenge_type doc whose keywords appear in the question."""
    if not question or not _db_available():
        return None
    try:
        ql = question.lower()
        for ct in _db.challenge_type.find({}):
            kws = ct.get("keywords") or []
            if isinstance(kws, str):
                kws = [kws]
            # Count keyword matches (substring, case-insensitive)
            match_count = 0
            for kw in kws:
                if not kw:
                    continue
                if str(kw).strip().lower() in ql:
                    match_count += 1
            # Require at least 2 keyword matches
            if match_count >= 2:
                return ct
        # No challenge_type met the threshold
        try:
            print("No match challenge type found")
        except Exception:
            pass
        return None
    except Exception:
        return None


def validate_question_and_get_model(question):
    """
    Validate question for keywords and retrieve corresponding model from MongoDB.
    Returns model document if keywords found, None otherwise.
    """
    if not question or not _db_available():
        return None
    
    # Prefer dynamic mapping via challenge_type keywords
    ct_doc = _find_challenge_type_for_question(question)
    if ct_doc:
        model_id = ct_doc.get("model_id")
        if model_id:
            try:
                return _db.model.find_one({"model_id": model_id})
            except Exception:
                return None
        # If challenge_type found but no model_id, fall through to active model
    # Fallback: active model or most recent
    try:
        active_model = _db.model.find_one({"is_active": True})
        if active_model:
            return active_model
        recent_model = _db.model.find_one({}, sort=[("_id", -1)])
        return recent_model
    except Exception:
        return None


def get_model_config(question):
    """
    Get model configuration based on question.
    If question contains 'two icons' or 'shapes', retrieve model from model collection.
    Otherwise, fall back to model_configs collection or default.
    """
    if not _db_available():
        return DEFAULT_MODEL_CONFIG
    
    # Validate question and get model if keywords found
    model_doc = validate_question_and_get_model(question)
    if model_doc:
        # Return config with model information
        return {
            'model_name': model_doc.get('model_name', 'default'),
            'model_id': model_doc.get('model_id'),
            'weights_id': str(model_doc.get('weights', '')),
            'results': model_doc.get('results', {}),
            'is_active': model_doc.get('is_active', False)
        }
    
    # Fallback to original logic: check model_configs collection
    try:
        config = _db.model_configs.find_one({"question": question}) or _db.model_configs.find_one({})
        return config or DEFAULT_MODEL_CONFIG
    except (errors.PyMongoError, Exception):
        return DEFAULT_MODEL_CONFIG

        
# =========================
# Model collection helpers
# =========================

def _fs_available():
    return _db_available() and _fs is not None


def upsert_model(model_id, model_name, weights_bytes, results, is_active=False):
    """
    Create or update a model document.

    - Stores weights in GridFS; saves reference ObjectId in `weights` field
    - Results is expected to be a dict containing metrics:
      {precision, recall, f1_score, mAP50, AP5095}
    - is_active: when True, marks other models as inactive
    """
    if not _db_available():
        return None

    weights_id = None
    if weights_bytes and _fs_available():
        try:
            # Overwrite by model_id-tagged filename for friendly lookup
            filename = f"{model_id}.pt"
            # Optionally remove previous file if exists
            prev = _db.model.find_one({"model_id": model_id})
            if prev and prev.get("weights"):
                try:
                    _fs.delete(prev["weights"])  # type: ignore[arg-type]
                except Exception:
                    pass
            
            # Store raw binary bytes directly in GridFS (no Base64 encoding)
            # GridFS handles binary data natively - it stores bytes directly without any encoding
            # Pass bytes directly or use BytesIO - both work, but BytesIO is more explicit for file-like interface
            from io import BytesIO
            
            # Ensure we have raw binary bytes (not string, not encoded)
            if isinstance(weights_bytes, str):
                # If somehow we got a string, encode it back to bytes
                # This shouldn't happen, but just in case
                weights_bytes = weights_bytes.encode('latin-1')
            
            # Create a file-like object from bytes for GridFS
            weights_stream = BytesIO(weights_bytes)
            
            # Put file in GridFS - stores raw binary data directly
            # GridFS does NOT encode to Base64 - it stores binary data as-is in chunks
            # The chunk_size parameter controls chunk size (default 255KB)
            # Smaller chunk_size = more chunks = more metadata overhead
            # Larger chunk_size = fewer chunks = less overhead but larger individual chunks
            weights_id = _fs.put(
                weights_stream, 
                filename=filename, 
                content_type="application/octet-stream"
                # Note: GridFS stores binary data natively - no encoding parameter needed
                # It will store the bytes directly without Base64 or any text encoding
            )
        except Exception as e:
            print(f"Error storing model weights in GridFS: {e}")
            import traceback
            traceback.print_exc()
            weights_id = None

    doc = {
        "model_id": model_id,
        "model_name": model_name,
        "results": results or {},
        "is_active": bool(is_active),
    }
    if weights_id is not None:
        doc["weights"] = weights_id

    try:
        # If activating this model, deactivate others first
        if is_active:
            _db.model.update_many({}, {"$set": {"is_active": False}})
        _db.model.update_one({"model_id": model_id}, {"$set": doc}, upsert=True)
        return _db.model.find_one({"model_id": model_id})
    except Exception:
        return None


def get_model_by_id(model_id):
    if not _db_available():
        return None
    try:
        return _db.model.find_one({"model_id": model_id})
    except Exception:
        return None


def list_models(limit=50):
    if not _db_available():
        return []
    try:
        cursor = _db.model.find({}).sort([("is_active", -1), ("model_name", 1)]).limit(int(limit))
        return list(cursor)
    except Exception:
        return []


def set_active_model(model_id):
    if not _db_available():
        return False
    try:
        _db.model.update_many({}, {"$set": {"is_active": False}})
        res = _db.model.update_one({"model_id": model_id}, {"$set": {"is_active": True}})
        return res.matched_count > 0
    except Exception:
        return False


def download_weights_bytes(model_id):
    """Return weights bytes for a model_id or None.
    
    Returns raw binary bytes directly from GridFS (no Base64 encoding).
    """
    if not _fs_available():
        return None
    try:
        doc = _db.model.find_one({"model_id": model_id})
        if not doc or not doc.get("weights"):
            return None
        file_id = doc["weights"]
        gridout = _fs.get(file_id)
        # Read raw binary bytes (GridFS returns binary data directly)
        weights_bytes = gridout.read()
        return weights_bytes
    except Exception as e:
        print(f"Error downloading model weights: {e}")
        return None


def get_preprocess_profile(preprocess_id):
    """Return preprocess_profile document by preprocess_id or None."""
    if not _db_available():
        return None
    try:
        return _db.preprocess_profile.find_one({"preprocess_id": preprocess_id})
    except Exception:
        return None


def get_preprocess_for_model(model_doc):
    """Return preprocess_profile document linked via model's preprocess_id, or None."""
    if not model_doc:
        return None
    preprocess_id = model_doc.get("preprocess_id")
    if not preprocess_id:
        return None
    return get_preprocess_profile(preprocess_id)


def get_postprocess_profile(postprocess_id):
    """Return postprocess_profile document by postprocess_id or None."""
    if not _db_available():
        return None
    try:
        return _db.postprocess_profile.find_one({"postprocess_id": postprocess_id})
    except Exception:
        return None


def get_postprocess_for_model(model_doc):
    """Return postprocess_profile document linked via model's postprocess_id, or None."""
    if not model_doc:
        return None
    postprocess_id = model_doc.get("postprocess_id")
    if not postprocess_id:
        return None
    return get_postprocess_profile(postprocess_id)


# ==============================
# Challenge Type collection API
# ==============================

def upsert_challenge_type(challenge_type_id, method, keywords, model_id=None):
    """
    Create or update a challenge_type document.

    Fields:
    - challenge_type_id: unique identifier (string)
    - method: "drag" or "click"
    - keywords: list of strings (e.g., ["two icons", "shapes"]) or a single string
    - model_id: optional relation to `model.model_id`
    """
    if not _db_available():
        return None

    if method not in ("drag", "click"):
        raise ValueError("method must be 'drag' or 'click'")

    if isinstance(keywords, str):
        keywords = [keywords]

    doc = {
        "challenge_type_id": challenge_type_id,
        "method": method,
        "keywords": keywords or [],
        "model_id": model_id,
    }

    try:
        _db.challenge_type.update_one(
            {"challenge_type_id": challenge_type_id},
            {"$set": doc},
            upsert=True,
        )
        return _db.challenge_type.find_one({"challenge_type_id": challenge_type_id})
    except Exception:
        return None


def get_challenge_type_by_id(challenge_type_id):
    if not _db_available():
        return None
    try:
        return _db.challenge_type.find_one({"challenge_type_id": challenge_type_id})
    except Exception:
        return None


def list_challenge_types(limit=100):
    if not _db_available():
        return []
    try:
        cursor = _db.challenge_type.find({}).sort("challenge_type_id", 1).limit(int(limit))
        return list(cursor)
    except Exception:
        return []


# ==========================
# Challenge collection API
# ==========================

def create_challenge(challenge_id, challenge_questions, challenge_img, challenge_type_id=None, timestamp=None):
    """
    Insert a new challenge document.

    Fields:
    - challenge_id: unique identifier (string)
    - challenge_questions: single string
    - challenge_img: raw bytes, base64 string, or any storable BSON type
    - challenge_type_id: relation to `challenge_type.challenge_type_id`
    - timestamp: optional datetime; defaults to utcnow
    """
    if not _db_available():
        return None

    # Normalize to string
    if isinstance(challenge_questions, list):
        challenge_questions = " ".join([q for q in challenge_questions if q])

    # If challenge_type_id not provided, try to derive from question text via keywords
    derived_type_id = None
    try:
        ct = _find_challenge_type_for_question(challenge_questions or "")
        if ct:
            derived_type_id = ct.get("challenge_type_id")
    except Exception:
        derived_type_id = None

    # Best-effort referential integrity: ensure provided or derived challenge_type exists
    effective_type_id = challenge_type_id or derived_type_id
    if effective_type_id:
        try:
            exists = _db.challenge_type.find_one({"challenge_type_id": effective_type_id})
            if not exists:
                raise ValueError("challenge_type_id does not exist")
        except Exception:
            raise

    doc = {
        "challenge_id": challenge_id,
        "challenge_questions": challenge_questions or "",
        "challenge_img": challenge_img,
        "challenge_type_id": effective_type_id,
        "timestamp": timestamp or datetime.utcnow(),
    }

    try:
        _db.challenge.update_one({"challenge_id": challenge_id}, {"$set": doc}, upsert=True)
        return _db.challenge.find_one({"challenge_id": challenge_id})
    except Exception:
        return None


def get_model_for_challenge(challenge_id):
    """Return the model document linked via the challenge's challenge_type.model_id."""
    if not _db_available():
        return None
    try:
        ch = _db.challenge.find_one({"challenge_id": challenge_id})
        if not ch:
            return None
        ct_id = ch.get("challenge_type_id")
        if not ct_id:
            return None
        ct = _db.challenge_type.find_one({"challenge_type_id": ct_id})
        if not ct:
            return None
        mid = ct.get("model_id")
        if not mid:
            return None
        return _db.model.find_one({"model_id": mid})
    except Exception:
        return None


def get_challenge_by_id(challenge_id):
    if not _db_available():
        return None
    try:
        return _db.challenge.find_one({"challenge_id": challenge_id})
    except Exception:
        return None


def list_challenges(limit=100):
    if not _db_available():
        return []
    try:
        cursor = _db.challenge.find({}).sort("timestamp", -1).limit(int(limit))
        return list(cursor)
    except Exception:
        return []


def ensure_indexes():
    """Create indexes for new collections. Safe to call multiple times."""
    if not _db_available():
        return False
    try:
        _db.challenge_type.create_index("challenge_type_id", unique=True)
        _db.challenge.create_index("challenge_id", unique=True)
        _db.challenge.create_index("challenge_type_id")
        _db.challenge.create_index("timestamp")
        # Inference collection indexes
        _db.inference.create_index("inference_id", unique=True)
        _db.inference.create_index("model_id")
        _db.inference.create_index("challenge_id")
        _db.inference.create_index("timestamp")
        _db.inference.create_index([("model_id", 1), ("timestamp", -1)])
        return True
    except Exception:
        return False


# ==========================
# Inference collection API
# ==========================

def create_inference(inference_id, inference_speed, detection_results, model_id, challenge_id, timestamp=None):
    """
    Insert a new inference document.
    
    Fields:
    - inference_id: unique identifier (string)
    - inference_speed: time taken for inference in seconds (float)
    - detection_results: detection results from model (list/dict)
      Format: [{'class': str, 'confidence': float, 'bbox': [x1, y1, x2, y2]}, ...]
      Or: {'error': str} or {'message': str, 'detections': []}
    - model_id: reference to model.model_id (string)
    - challenge_id: reference to challenge.challenge_id (string)
    - timestamp: optional datetime; defaults to utcnow
    """
    if not _db_available():
        return None
    
    doc = {
        "inference_id": inference_id,
        "inference_speed": float(inference_speed),
        "detection_results": detection_results,
        "model_id": model_id,
        "challenge_id": challenge_id,
        "timestamp": timestamp or datetime.utcnow(),
    }
    
    try:
        _db.inference.update_one({"inference_id": inference_id}, {"$set": doc}, upsert=True)
        return _db.inference.find_one({"inference_id": inference_id})
    except Exception:
        return None


def get_inference_by_id(inference_id):
    """Return inference document by inference_id or None."""
    if not _db_available():
        return None
    try:
        return _db.inference.find_one({"inference_id": inference_id})
    except Exception:
        return None


def list_inferences(limit=100, model_id=None, challenge_id=None):
    """
    List inference documents, optionally filtered by model_id or challenge_id.
    
    Args:
        limit: maximum number of results to return
        model_id: optional filter by model_id
        challenge_id: optional filter by challenge_id
    
    Returns:
        List of inference documents, sorted by timestamp (most recent first)
    """
    if not _db_available():
        return []
    
    try:
        query = {}
        if model_id:
            query["model_id"] = model_id
        if challenge_id:
            query["challenge_id"] = challenge_id
        
        cursor = _db.inference.find(query).sort("timestamp", -1).limit(int(limit))
        return list(cursor)
    except Exception:
        return []


def get_inferences_by_challenge(challenge_id, limit=100):
    """Return all inference documents for a specific challenge_id."""
    return list_inferences(limit=limit, challenge_id=challenge_id)


def get_inferences_by_model(model_id, limit=100):
    """Return all inference documents for a specific model_id."""
    return list_inferences(limit=limit, model_id=model_id)


# ==========================
# Activity Log collection API
# ==========================

def create_activity_log(log_id, timestamp=None, challenge_id=None, model_id=None, inference_id=None):
    """
    Insert a new activity_log document.
    
    Fields:
    - log_id: unique identifier (string)
    - timestamp: datetime (defaults to utcnow)
    - challenge_id: string reference to challenge.challenge_id
    - model_id: string reference to model.model_id
    - inference_id: string reference to inference.inference_id
    """
    if not _db_available():
        return None
    
    doc = {
        "log_id": log_id,
        "timestamp": timestamp or datetime.utcnow(),
        "challenge_id": challenge_id,
        "model_id": model_id,
        "inference_id": inference_id,
    }
    try:
        _db.activity_log.update_one({"log_id": log_id}, {"$set": doc}, upsert=True)
        return _db.activity_log.find_one({"log_id": log_id})
    except Exception:
        return None