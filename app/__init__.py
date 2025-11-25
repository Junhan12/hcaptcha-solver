"""Top-level package for the hCaptcha solver service.

Exposes:
- flask_app: the Flask application instance from api_gateway
- create_app(): simple factory returning the Flask app
- solve_captcha: core solver function
- get_model_config: database helper
"""

from .solver import solve_captcha
from .database import get_model_config, validate_question_and_get_model
from .helper import decompress_image_to_base64
from .database import (
    upsert_model,
    get_model_by_id,
    list_models,
)

# Keep it None until create_app() is called.
flask_app = None  # type: ignore


def create_app():
    """Return the Flask app instance.

    This provides a minimal factory for WSGI servers (e.g., gunicorn) or
    external callers that prefer `from app import create_app`.
    """
    global flask_app
    if flask_app is None:
        from .api_gateway import app as _app
        flask_app = _app
    return flask_app


__all__ = [
    "flask_app",
    "create_app",
    "solve_captcha",
    "get_model_config",
    "validate_question_and_get_model",
    "decompress_image_to_base64",
    "upsert_model",
    "get_model_by_id",
    "list_models",
]

